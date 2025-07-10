import argparse
import multiprocessing
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool
from tqdm import tqdm

# 将项目根目录添加到Python路径中
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.graph.structural_graph import BeamAsNodeGraph
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel, StructuralParameters

# 全局变量，用于工作进程
section_library_global = None


def init_worker():
    """为多进程池初始化工作进程，每个进程创建一个SectionLibrary实例。"""
    global section_library_global
    section_library_global = SectionLibrary()


def generate_graph_worker(row_dict: dict):
    """
    工作函数，用于从一行数据生成单个图。
    接收一个DataFrame行的字典表示。
    """
    try:
        params = StructuralParameters(
            length=row_dict["length"],
            width=3.6,
            height=3.3,
            main_beam_spacing=row_dict["main_beam_spacing"],
            secondary_beam_spacing=row_dict["main_beam_spacing"],
            brace_spacing=row_dict["brace_spacing"],
            frame_section_index=int(row_dict["frame_section_index"]),
            main_beam_section_index=int(row_dict["main_beam_section_index"]),
            secondary_beam_section_index=int(row_dict["secondary_beam_section_index"]),
            brace_section_index=int(row_dict["brace_section_index"]),
            pressure=row_dict["pressure"],
            windows=[],
        )

        model = StructuralModel(params, section_library_global)
        model.generate_grid_mesh()
        graph_converter = BeamAsNodeGraph(model)
        pyg_data = graph_converter.to_pyg_data({})

        targets = [row_dict["max_stress"], row_dict["max_displacement"]]

        return pyg_data, targets
    except Exception:
        return None


# 1. 定义新的高效GNN回归模型
class EfficientGNNRegressor(nn.Module):
    """
    基于GATv2的高效GNN模型，用于结构响应预测。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        # 使用GATv2卷积层，性能更好
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=True, add_self_loops=False)
        self.bn1 = BatchNorm(hidden_dim * num_heads)

        self.conv2 = GATv2Conv(
            hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, add_self_loops=False
        )
        self.bn2 = BatchNorm(hidden_dim * num_heads)

        # 最后一层注意力头合并
        self.conv3 = GATv2Conv(
            hidden_dim * num_heads, hidden_dim, heads=1, concat=False, add_self_loops=False
        )
        self.bn3 = BatchNorm(hidden_dim)

        # 回归头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT层
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # 全局池化层，得到图级别的嵌入
        graph_embedding = global_mean_pool(x, batch)

        # MLP回归头
        return self.mlp(graph_embedding)


def preprocess_and_cache_graphs(csv_path: str, cache_path: str) -> tuple[list[Data], torch.Tensor]:
    """从CSV生成图数据并缓存，如果缓存存在则直接加载。"""
    if os.path.exists(cache_path):
        print(f"从缓存加载已处理的图数据: {cache_path}")
        return torch.load(cache_path)

    print("未找到缓存，正在从CSV生成图数据...")
    df = pd.read_csv(csv_path)

    data_dicts = df.to_dict("records")
    results = []
    failed_count = 0

    with multiprocessing.Pool(initializer=init_worker) as pool:
        with tqdm(total=len(data_dicts), desc="使用多进程生成图") as pbar:
            for result in pool.imap_unordered(generate_graph_worker, data_dicts):
                if result is not None:
                    results.append(result)
                else:
                    failed_count += 1
                pbar.update(1)

    if not results:
        raise ValueError("未能成功生成任何图数据。请检查错误。")

    if failed_count > 0:
        print(f"警告: {failed_count} 个样本在图生成过程中失败。")

    graph_list, target_list = zip(*results)
    targets_tensor = torch.FloatTensor(target_list)

    print(f"成功生成 {len(graph_list)} / {len(data_dicts)} 个图。")
    print(f"缓存图数据到: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save((list(graph_list), targets_tensor), cache_path)

    return list(graph_list), targets_tensor


def train_gnn_model(
    dataset_csv_path: str,
    processed_data_cache_path: str,
    model_save_dir: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    resume_from_checkpoint: bool = False,
):
    """训练GNN代理模型。"""

    # --- 0. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # --- 1. 数据准备 ---
    graph_list, targets = preprocess_and_cache_graphs(dataset_csv_path, processed_data_cache_path)

    # 数据集划分
    indices = list(range(len(graph_list)))
    train_val_indices, test_indices, y_train_val, y_test = train_test_split(
        indices, targets, test_size=0.1, random_state=42
    )
    train_indices, val_indices, y_train, y_val = train_test_split(
        train_val_indices, y_train_val, test_size=(2 / 9), random_state=42
    )

    X_train = [graph_list[i] for i in train_indices]
    X_val = [graph_list[i] for i in val_indices]
    X_test = [graph_list[i] for i in test_indices]

    # --- 2. 特征和目标标准化 ---
    # 节点特征标准化
    all_train_node_features = torch.cat([data.x for data in X_train], dim=0)
    node_scaler = StandardScaler()
    node_scaler.fit(all_train_node_features.numpy())

    for data in graph_list:  # 在所有图上应用scaler
        data.x = torch.from_numpy(node_scaler.transform(data.x.numpy())).float()

    # 目标值标准化
    target_scaler = StandardScaler()
    y_train_scaled = torch.from_numpy(target_scaler.fit_transform(y_train.numpy())).float()
    y_val_scaled = torch.from_numpy(target_scaler.transform(y_val.numpy())).float()
    y_test_scaled = torch.from_numpy(target_scaler.transform(y_test.numpy())).float()

    # 关键修复：将目标y附加到每个图对象，并确保其为2D张量 [1, num_targets]
    for i, data in enumerate(X_train):
        data.y = y_train_scaled[i].unsqueeze(0)
    for i, data in enumerate(X_val):
        data.y = y_val_scaled[i].unsqueeze(0)
    for i, data in enumerate(X_test):
        data.y = y_test_scaled[i].unsqueeze(0)

    # 创建 DataLoader
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=batch_size)  # 验证集不需要打乱
    test_loader = DataLoader(X_test, batch_size=batch_size)

    # --- 3. 模型初始化 ---
    # 从第一个图的节点特征推断输入维度
    input_dim = X_train[0].x.size(1)
    output_dim = y_train.size(1)

    # 使用新的高效GNN模型
    model = EfficientGNNRegressor(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_heads=4)
    model.to(device)  # 将模型移动到GPU

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. 从检查点加载（如果需要） ---
    start_epoch = 0
    best_val_loss = float("inf")
    latest_checkpoint_path = os.path.join(model_save_dir, "latest_checkpoint.pth")

    if resume_from_checkpoint:
        if os.path.exists(latest_checkpoint_path):
            print(f"从检查点恢复: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"恢复成功。将从 epoch {start_epoch} 开始训练。")
        else:
            print("警告: 未找到检查点文件。将从头开始训练。")

    # --- 5. 训练循环 ---
    os.makedirs(model_save_dir, exist_ok=True)
    log_interval = max(1, len(train_loader) // 10)  # 大约每个epoch记录10次

    print("\n开始训练GNN代理模型...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0
        # 修正后的训练循环
        for i, data_batch in enumerate(train_loader):
            data_batch = data_batch.to(device)  # 将批次数据移动到GPU
            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = criterion(outputs, data_batch.y)  # 使用来自批次的正确y
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            if (i + 1) % log_interval == 0:
                print(
                    f"  Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}, Batch Loss: {loss.item():.6f}"
                )

        avg_train_loss = epoch_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data_batch in val_loader:
                data_batch = data_batch.to(device)  # 将批次数据移动到GPU
                val_outputs = model(data_batch)
                val_loss = criterion(val_outputs, data_batch.y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "efficient_gnn_surrogate_model.pth"))
            print(f"  ** 新的最佳验证损失: {best_val_loss:.6f}。已保存最佳模型。 **")

        # 保存最新的检查点用于恢复
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            latest_checkpoint_path,
        )

    print("\n训练完成!")

    # --- 6. 测试集评估 ---
    print("在测试集上评估最佳模型...")
    # model.state_dict()包含CPU张量，所以我们加载到CPU模型，然后移动到设备
    best_model_state = torch.load(os.path.join(model_save_dir, "efficient_gnn_surrogate_model.pth"))
    model.load_state_dict(best_model_state)
    model.to(device)  # 确保模型在正确的设备上
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data_batch in test_loader:
            data_batch = data_batch.to(device)  # 将批次数据移动到GPU
            test_outputs = model(data_batch)
            test_loss = criterion(test_outputs, data_batch.y)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"测试集损失 (MSE): {avg_test_loss:.6f}")

    # --- 7. 保存Scalers ---
    joblib.dump(node_scaler, os.path.join(model_save_dir, "gnn_node_scaler.joblib"))
    joblib.dump(target_scaler, os.path.join(model_save_dir, "gnn_target_scaler.joblib"))
    print(f"GNN模型和scalers已保存到 '{model_save_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练GNN代理模型")
    parser.add_argument("--resume", action="store_true", help="从最新的检查点恢复训练")
    args = parser.parse_args()

    train_gnn_model(
        dataset_csv_path="results/surrogate_model_dataset/dataset_0709.csv",
        processed_data_cache_path="results/surrogate_model_dataset/processed_graphs.pt",
        model_save_dir="results/gnn_surrogate_model",
        num_epochs=100,
        batch_size=32,
        resume_from_checkpoint=args.resume,
    )
