import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 1. 定义神经网络模型
class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def plot_training_history(history, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train_model(data_path: str, model_save_dir: str, num_epochs: int = 200, batch_size: int = 64):
    """
    加载数据，训练代理模型，并保存模型和scalers。

    Args:
        data_path (str): 数据集CSV文件的路径。
        model_save_dir (str): 保存训练好的模型和scalers的目录。
        num_epochs (int): 训练周期数。
        batch_size (int): 批量大小。
    """
    # --- 2. 加载和预处理数据 ---
    if not os.path.exists(data_path):
        print(f"错误：找不到数据集文件 '{data_path}'。")
        print("请先运行 'generate_dataset.py' 来生成数据。")
        return
    os.makedirs(model_save_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # 定义输入特征和输出目标
    features = [
        "length",
        "main_beam_spacing",
        "brace_spacing",
        "frame_section_index",
        "main_beam_section_index",
        "secondary_beam_section_index",
        "brace_section_index",
        "pressure",
    ]
    targets = ["max_stress", "max_displacement"]

    X = df[features].values
    y = df[targets].values

    # 划分训练集、验证集和测试集 (70-20-10)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(2 / 9), random_state=42
    )  # 2/9 of 90% is 20%

    # 标准化数据
    # 这是至关重要的一步，可以使训练更稳定
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 3. 初始化模型、损失和优化器 ---
    input_dim = len(features)
    output_dim = len(targets)
    model = SurrogateModel(input_dim, output_dim)
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. 训练模型 ---
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("开始训练代理模型...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # 在验证集上评估
        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                history["val_loss"].append(val_loss.item())

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Validation Loss: {val_loss.item():.6f}"
                )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(model_save_dir, "surrogate_model.pth"))

    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # --- 5. 在测试集上评估最佳模型 ---
    print("\n在测试集上评估最佳模型...")
    best_model = SurrogateModel(input_dim, output_dim)
    best_model.load_state_dict(torch.load(os.path.join(model_save_dir, "surrogate_model.pth")))
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f"测试集损失 (MSE): {test_loss.item():.6f}")

    # --- 6. 保存Scalers和训练历史图 ---
    joblib.dump(scaler_X, os.path.join(model_save_dir, "input_scaler.joblib"))
    joblib.dump(scaler_y, os.path.join(model_save_dir, "output_scaler.joblib"))
    print(f"模型和scalers已保存到 '{model_save_dir}'")

    plot_save_path = os.path.join(model_save_dir, "training_history.png")
    plot_training_history(history, plot_save_path)
    print(f"训练历史图已保存到 '{plot_save_path}'")


def predict_with_model(input_data: dict, model_dir: str):
    """
    使用已训练的模型和scalers进行预测。

    Args:
        input_data (dict): 包含输入特征的字典。
        model_dir (str): 存放模型和scalers的目录。
    """
    # 加载模型和scalers
    model_path = os.path.join(model_dir, "surrogate_model.pth")
    scaler_X_path = os.path.join(model_dir, "input_scaler.joblib")
    scaler_y_path = os.path.join(model_dir, "output_scaler.joblib")

    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
        print("错误：找不到模型或scaler文件。请先运行训练。")
        return

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # 从字典推断维度并创建模型
    input_dim = len(input_data)
    # 从scaler_y推断输出维度
    output_dim = scaler_y.n_features_in_
    model = SurrogateModel(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 准备输入数据
    # 将输入字典转换为与训练时相同顺序的numpy数组
    features_order = [
        "length",
        "main_beam_spacing",
        "brace_spacing",
        "frame_section_index",
        "main_beam_section_index",
        "secondary_beam_section_index",
        "brace_section_index",
        "pressure",
    ]
    input_array = np.array([input_data[feat] for feat in features_order]).reshape(1, -1)

    # 缩放输入数据并转换为张量
    input_scaled = scaler_X.transform(input_array)
    input_tensor = torch.FloatTensor(input_scaled)

    # 进行预测
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
        # 将预测结果反向缩放回原始尺度
        prediction = scaler_y.inverse_transform(prediction_scaled.numpy())

    print("\n--- 预测结果 ---")
    print(f"输入参数: {input_data}")
    print(f"预测的最大应力: {prediction[0][0]:.2f} MPa")
    print(f"预测的最大位移: {prediction[0][1] * 1000:.2f} mm")


if __name__ == "__main__":
    dataset_path = "results/surrogate_model_dataset/dataset_0709.csv"
    save_dir = "results/surrogate_model"

    # 训练模型
    # train_model(data_path=dataset_path, model_save_dir=save_dir)

    # 使用训练好的模型进行预测的示例
    print("\n--- 运行预测示例 ---")
    example_input = {
        "length": 10.0,
        "main_beam_spacing": 1.2,
        "brace_spacing": 0.6,
        "frame_section_index": 3,
        "main_beam_section_index": 4,
        "secondary_beam_section_index": 5,
        "brace_section_index": 6,
        "pressure": 30,
    }
    predict_with_model(input_data=example_input, model_dir=save_dir)
