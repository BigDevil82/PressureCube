import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
from matplotlib import pyplot as plt


def run_plate_analysis():
    """
    通过OpenSees建立一个1mx1m的四面固支板三维模型，
    材料采用钢，单元采用ShellMITC4，板上施加30kN/m2的均布向下荷载，
    输出板的应力。
    """
    # 1. 初始化模型
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # 2. 定义模型参数
    # 板的尺寸
    L = 1.0  # 板的边长 (m)
    h = 0.005  # 板的厚度 (m)，此为假设值
    t = h

    # 材料属性 (钢)
    E = 2.06e11  # 杨氏模量 (Pa = N/m^2) -> 2.0e8 kN/m^2
    nu = 0.3  # 泊松比

    # 荷载
    load_mag = -30.0e3  # 均布荷载 (kN/m^2), 负号表示向下

    # 网格划分
    nx = 61  # x方向节点数
    ny = 61  # y方向节点数

    # 3. 定义节点
    node_tags = np.arange(1, nx * ny + 1).reshape(nx, ny)
    for i in range(nx):
        for j in range(ny):
            x = i * L / (nx - 1)
            y = j * L / (ny - 1)
            z = 0.0
            ops.node(int(node_tags[i, j]), x, y, z)

    # 4. 定义边界条件 (四边固支)
    # boundary_nodes = [(0, 0), (0, ny - 1), (nx - 1, 0), (nx - 1, ny - 1)]
    # for i, j in boundary_nodes:
    #     ops.fix(int(node_tags[i, j]), 1, 1, 1, 1, 1, 1)
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                ops.fix(int(node_tags[i, j]), 1, 1, 1, 1, 1, 1)

    # 5. 定义材料和截面
    sec_tag = 1
    ops.section("ElasticMembranePlateSection", sec_tag, E, nu, h, 7850)

    # 6. 定义单元 (ShellMITC4)
    ele_tag = 1
    for i in range(nx - 1):
        for j in range(ny - 1):
            n1 = int(node_tags[i, j])
            n2 = int(node_tags[i + 1, j])
            n3 = int(node_tags[i + 1, j + 1])
            n4 = int(node_tags[i, j + 1])
            ops.element("ShellMITC4", ele_tag, n1, n2, n3, n4, sec_tag)
            ele_tag += 1

    # 7. 定义荷载
    ts_tag = 1
    ops.timeSeries("Linear", ts_tag)
    pattern_tag = 1
    ops.pattern("Plain", pattern_tag, ts_tag)

    # 将面荷载等效为节点荷载
    # 每个小单元的面积
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    ele_area = dx * dy
    # 每个小单元上的总力
    ele_force = load_mag * ele_area
    # 分配到每个节点的力 (Z方向)
    node_force = ele_force / 4.0

    # 遍历所有单元，将力施加到其节点上
    # 使用一个字典来累加每个节点上的荷载
    nodal_loads = {}
    for i in range(nx - 1):
        for j in range(ny - 1):
            n1 = int(node_tags[i, j])
            n2 = int(node_tags[i + 1, j])
            n3 = int(node_tags[i + 1, j + 1])
            n4 = int(node_tags[i, j + 1])

            ele_nodes = [n1, n2, n3, n4]
            for node_tag in ele_nodes:
                if node_tag not in nodal_loads:
                    nodal_loads[node_tag] = 0.0
                nodal_loads[node_tag] += node_force

    # 将累加后的节点荷载施加到模型
    for node_tag, force_z in nodal_loads.items():
        ops.load(node_tag, 0.0, 0.0, force_z, 0.0, 0.0, 0.0)

    # 8. 定义分析
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")

    # 9. 运行分析
    ops.analyze(1)

    print("Analysis complete.")

    # 10. 获取并输出结果
    # 找到中心单元
    center_i = (nx - 1) // 2
    center_j = (ny - 1) // 2
    center_ele_tag = center_i * (ny - 1) + center_j + 1

    # 获取应力
    # 对于ShellMITC4, 'stress' 参数返回在每个积分点上的应力
    # 对于弹性截面，通常有4个积分点（高斯点）在面内，以及在厚度方向上的上下两个点
    # 返回值为 [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx] for each integration point
    # 这里我们只关心面内应力 sig_xx, sig_yy, sig_xy
    # 我们将获取板顶部和底部的应力

    ops.reactions()

    # 获取节点位移最大值
    max_abs_disp = 0
    max_abs_disp_node_tag = None
    for node_tag in range(1, nx * ny + 1):
        disp = ops.nodeDisp(node_tag)
        abs_disp = np.sqrt(disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2)
        if abs_disp > max_abs_disp:
            max_abs_disp = abs_disp
            max_abs_disp_node_tag = node_tag
    print(f"Max absolute displacement: {max_abs_disp*1e3:.4f} mm")
    print(f"Max absolute displacement node tag: {max_abs_disp_node_tag}")

    print(f"\nStresses at center element (tag {center_ele_tag}):")

    # 获取底部纤维的应力
    stress_bottom = ops.eleResponse(center_ele_tag, "stresses")
    print("p11, p22, p12, m11, m22, m12, q1, q2")
    for i in range(4):
        # 格式化打印，控制间距
        p11, p22, p12, m11, m22, m12, q1, q2 = stress_bottom[8 * i : 8 * (i + 1)]
        print(f"{p11:.4f} {p22:.4f} {p12:.4f} {m11:.4f} {m22:.4f} {m12:.4f} {q1:.4f} {q2:.4f}")
    print()

    # 2. 初始化存储最大真实应力的变量
    max_true_stresses = {
        "sigma_11": {"value": 0, "eleTag": None, "gaussPoint": None, "surface": None},
        "sigma_22": {"value": 0, "eleTag": None, "gaussPoint": None, "surface": None},
        "sigma_12": {"value": 0, "eleTag": None, "gaussPoint": None, "surface": None},
        "tau_xz": {"value": 0, "eleTag": None, "gaussPoint": None},
        "tau_yz": {"value": 0, "eleTag": None, "gaussPoint": None},
    }

    # 3. 遍历所有 ShellMITC4 单元
    total_ele = (nx - 1) * (ny - 1)
    all_stresses = []
    for eleTag in range(1, total_ele + 1):

        # 获取广义力/力矩响应 (32 值)
        stresses = ops.eleResponse(eleTag, "stresses")
        if not stresses:
            continue

        # 按高斯点分组 (4 Gauss points × 8 components)
        stresses = np.array(stresses).reshape(4, 8)
        all_stresses.append(stresses)

        # 遍历每个高斯点
        for gaussPoint in range(4):
            p11, p22, p12, m11, m22, m12, q1, q2 = stresses[gaussPoint]

            # 计算膜应力
            sigma_11_membrane = p11 / t
            sigma_22_membrane = p22 / t
            sigma_12_membrane = p12 / t

            # 计算弯曲应力（上表面和下表面）
            sigma_11_bending_top = 6 * m11 / t**2
            sigma_11_bending_bottom = -6 * m11 / t**2
            sigma_22_bending_top = 6 * m22 / t**2
            sigma_22_bending_bottom = -6 * m22 / t**2
            sigma_12_bending_top = 6 * m12 / t**2
            sigma_12_bending_bottom = -6 * m12 / t**2

            # 总应力（上表面和下表面）
            sigma_11_top = sigma_11_membrane + sigma_11_bending_top
            sigma_11_bottom = sigma_11_membrane + sigma_11_bending_bottom
            sigma_22_top = sigma_22_membrane + sigma_22_bending_top
            sigma_22_bottom = sigma_22_membrane + sigma_22_bending_bottom
            sigma_12_top = sigma_12_membrane + sigma_12_bending_top
            sigma_12_bottom = sigma_12_membrane + sigma_12_bending_bottom

            # 横向剪应力
            tau_xz = q1 / t
            tau_yz = q2 / t

            # 更新最大值（绝对值比较）
            stresses_to_check = {
                "sigma_11": [(sigma_11_top, "top"), (sigma_11_bottom, "bottom")],
                "sigma_22": [(sigma_22_top, "top"), (sigma_22_bottom, "bottom")],
                "sigma_12": [(sigma_12_top, "top"), (sigma_12_bottom, "bottom")],
                "tau_xz": [(tau_xz, None)],
                "tau_yz": [(tau_yz, None)],
            }

            for component, values in stresses_to_check.items():
                for val, surface in values:
                    if abs(val) > abs(max_true_stresses[component]["value"]):
                        max_true_stresses[component]["value"] = val
                        max_true_stresses[component]["eleTag"] = eleTag
                        max_true_stresses[component]["gaussPoint"] = gaussPoint + 1
                        if surface:
                            max_true_stresses[component]["surface"] = surface

    # 4. 打印结果
    print("Maximum true stresses across all ShellMITC4 elements:")
    for component, data in max_true_stresses.items():
        print(f"{component}:")
        print(f"  Max value = {data['value']/1e6:.4f} MPa")
        print(f"  Element ID = {data['eleTag']}")
        print(f"  Gauss Point = {data['gaussPoint']}")
        if "surface" in data:
            print(f"  Surface = {data['surface']}")
        print("------")

    all_stresses = np.concatenate(all_stresses, axis=0)
    max_force = np.max(all_stresses, axis=0)
    min_force = np.min(all_stresses, axis=0)
    print("max force:")
    print((max_force / 1e3).tolist(), "kN")
    print((min_force / 1e3).tolist(), "kN")
    max_stress = np.zeros_like(max_force)
    min_stress = np.zeros_like(max_force)
    for i in range(max_force.shape[0]):
        if 3 < i < 6:
            max_stress[i] = 6 * max_force[i] / t**2
            min_stress[i] = 6 * min_force[i] / t**2
        else:
            max_stress[i] = max_force[i] / t
            min_stress[i] = min_force[i] / t
    print("max stress:")
    print((max_stress / 1e6).tolist(), "MPa")
    print((min_stress / 1e6).tolist(), "MPa")

    # plot stress with opsvis
    # opsv.plot_stress("sxx")
    # opsv.plot_stress("syy")
    # opsv.plot_stress("sxy")
    # plt.axis("equal")
    # plt.show()


if __name__ == "__main__":
    run_plate_analysis()
    ops.wipe()
