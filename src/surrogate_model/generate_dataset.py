import multiprocessing
import os
import sys
from traceback import print_exc

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.structure.mesh_generator import MeshGenerator, MeshParameters
from src.structure.random_structure import RandomStructure
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel


def generate_single_sample(_):
    """生成单个结构样本并对其进行分析。"""
    try:
        # 每个进程都需要自己的SectionLibrary和RandomStructure实例
        # 以避免状态共享并确保不同的随机数。
        section_library = SectionLibrary()
        random_structure = RandomStructure(section_library)

        # 1. 生成随机参数（不生成窗户）
        params = random_structure.generate_random_parameters(generate_windows=False)

        # 2. 创建结构模型和网格
        model = StructuralModel(params, section_library)
        model.generate_grid_mesh()

        mesh_params = MeshParameters(max_grid_size=0.2)
        mesh_generator = MeshGenerator(model, mesh_params)
        mesh_generator.generate_mesh()
        meshed_model = mesh_generator.meshed_model

        # 3. 运行有限元分析
        analyzer = OpenSeesAnalyzer()
        analyzer.setup_opensees_model(meshed_model)
        # 压力单位从kPa转换为Pa
        analyzer.apply_pressure_loads(internal_pressure_pa=params.pressure * 1000.0)
        analyzer.run_static_analysis()
        max_indicators = analyzer.get_max_indicators()

        # 4. 存储结果
        result_item = {
            "length": params.length,
            "main_beam_spacing": params.main_beam_spacing,
            "brace_spacing": params.brace_spacing,
            "frame_section_index": params.frame_section_index,
            "main_beam_section_index": params.main_beam_section_index,
            "secondary_beam_section_index": params.secondary_beam_section_index,
            "brace_section_index": params.brace_section_index,
            "pressure": params.pressure,
            "max_stress": max_indicators["max_abs_stress"],
            "max_displacement": max_indicators["max_abs_displacement"],
        }
        return result_item

    except Exception:
        # 打印异常但不要使整个池崩溃
        print("生成样本时出错。正在跳过。")
        print_exc()
        return None


def generate_surrogate_model_dataset(num_samples: int, output_path: str):
    """
    通过在多个进程上对随机生成的结构进行有限元分析，为代理模型生成数据集。

    Args:
        num_samples (int): 要生成的样本数量。
        output_path (str): 保存生成的数据集（CSV文件）的路径。
    """
    results = []

    # 为进程池使用上下文管理器
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # 使用imap_unordered来在结果就绪时获取它们
        # 并用tqdm包装以显示进度条
        with tqdm(total=num_samples, desc="使用多进程生成数据集") as pbar:
            for result in pool.imap_unordered(generate_single_sample, range(num_samples)):
                if result is not None:
                    results.append(result)
                pbar.update(1)

    # 5. 将数据集保存为CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n已将 {len(results)} 个样本的数据集保存到 {output_path}")
    else:
        print("\n没有成功生成任何样本。")


if __name__ == "__main__":
    # 确保输出目录存在
    output_dir = "src/data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_csv_path = os.path.join(output_dir, "fem_dataset.csv")

    # # 生成10000个样本作为示例
    # generate_surrogate_model_dataset(num_samples=10000, output_path=output_csv_path)
    for i in range(10):
        data = generate_single_sample(i)
        print(data)
