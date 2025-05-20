import os

import yaml

from src.structure.structural_model import StructuralModel, StructuralParameters, Window


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            "structural_params": {
                "length": 7.2,
                "width": 3.6,
                "height": 3.3,
                "main_beam_spacing": 1.2,
                "secondary_beam_spacing": 1.2,
                "brace_spacing": 0.6,
                "windows": [
                    {"y_position": 3.6, "width": 1.2, "height": 1.5, "bottom_height": 1.0, "on_face": "front"}
                ],
            },
            "rl_params": {
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 64,
                "buffer_size": 10000,
                "update_interval": 5,
            },
            "gnn_params": {"in_channels": 10, "hidden_channels": 64, "out_channels": 32},
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_structural_model(config):
    """创建结构模型"""
    params_dict = config["structural_params"]

    # 处理窗户参数
    windows = []
    if "windows" in params_dict:
        for window_dict in params_dict["windows"]:
            windows.append(Window(**window_dict))

    # 创建结构参数对象
    params = StructuralParameters(
        length=params_dict["length"],
        width=params_dict["width"],
        height=params_dict["height"],
        main_beam_spacing=params_dict["main_beam_spacing"],
        secondary_beam_spacing=params_dict["secondary_beam_spacing"],
        brace_spacing=params_dict["brace_spacing"],
        windows=windows,
    )

    # 创建结构模型并生成网格
    model = StructuralModel(params)

    # 使用网格化建模方法
    model.generate_grid_mesh()

    return model
