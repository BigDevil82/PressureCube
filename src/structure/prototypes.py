import copy


def get_prototype_configs(base_config: dict):
    """
    加载基础配置并生成三个具有不同长度的原型结构配置。

    Args:
        config_path (str): 基础配置文件的路径。

    Returns:
        list: 包含三个原型配置字典的列表。
    """
    prototypes = []

    lengths = [9.6]

    for length in lengths:
        proto_config = copy.deepcopy(base_config)
        proto_config["structural_params"]["length"] = length
        # 为了区分，我们可以给每个原型起个名字
        proto_config["name"] = f"prototype_{length}m"
        prototypes.append(proto_config)

    return prototypes


if __name__ == "__main__":
    # 测试函数
    prototype_configs = get_prototype_configs()
    for config in prototype_configs:
        print(f"Loaded prototype: {config['name']}")
        print(f"  Length: {config['structural_params']['length']}m")
        print("-" * 20)
