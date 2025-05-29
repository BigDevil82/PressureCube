#!/usr/bin/env python3
"""
测试OpenSees钢板建模功能
演示增压建筑钢板蒙皮的分析流程
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.utils.model_utils import create_structural_model, load_config


def test_opensees_plate_modeling():
    """测试OpenSees钢板建模功能"""

    print("=" * 60)
    print("OpenSees增压建筑钢板蒙皮分析测试")
    print("=" * 60)

    # 加载配置和创建结构模型
    try:
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"错误：配置文件 {config_path} 不存在")
            return False

        config = load_config(config_path)
        model = create_structural_model(config)

        print(f"结构模型创建成功:")
        print(f"  - 节点数量: {len(model.points)}")
        print(f"  - 梁单元数量: {len(model.beams)}")
        print(f"  - 钢板数量: {len(model.plates) if model.plates else 0}")

        if not model.plates:
            print("警告：模型中没有钢板，请先运行钢板生成功能")
            return False

    except Exception as e:
        print(f"模型创建失败: {e}")
        return False

    # 创建OpenSees分析器
    analyzer = OpenSeesAnalyzer()

    # 测试参数
    test_scenarios = [
        # {
        #     "name": "标准增压分析",
        #     "plate_thickness_mm": 10.0,
        #     "internal_pressure_pa": 2000.0,  # 2 kPa
        #     "include_pressure": True,
        #     "output_dir": "results/opensees_plate_test_standard",
        # },
        {
            "name": "高压增压分析",
            "plate_thickness_mm": 10.0,
            "internal_pressure_pa": 30000.0,  # 30 kPa
            "include_pressure": True,
            "output_dir": "results/opensees_plate_test_high_pressure",
        },
        # {
        #     "name": "仅重力荷载分析",
        #     "plate_thickness_mm": 10.0,
        #     "internal_pressure_pa": 0.0,
        #     "include_pressure": False,
        #     "output_dir": "results/opensees_plate_test_gravity_only",
        # },
    ]

    # 执行测试场景
    all_results = {}

    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*40}")
        print(f"测试场景 {i+1}: {scenario['name']}")
        print(f"{'='*40}")

        try:
            # 执行完整分析
            analyzer.perform_full_analysis_and_save(
                model=model,
                num_modal_modes=6,
                output_dir=scenario["output_dir"],
                plate_thickness_mm=scenario["plate_thickness_mm"],
                internal_pressure_pa=scenario["internal_pressure_pa"],
                include_pressure_loads=scenario["include_pressure"],
            )

            # 读取分析结果
            results_file = os.path.join(scenario["output_dir"], "static_analysis_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                all_results[scenario["name"]] = results

                # 打印关键结果
                if "plate_analysis" in results:
                    plate_analysis = results["plate_analysis"]
                    if "plate_deformation_analysis" in plate_analysis:
                        deform_data = plate_analysis["plate_deformation_analysis"]
                        if "performance_indicators" in deform_data:
                            indicators = deform_data["performance_indicators"]
                            print(f"  最大钢板位移: {indicators.get('max_plate_displacement_mm', 0):.4f} mm")
                            print(
                                f"  平均钢板位移: {indicators.get('average_plate_displacement_mm', 0):.4f} mm"
                            )
                            print(f"  位移标准差: {indicators.get('displacement_std_mm', 0):.4f} mm")

                print(f"  分析结果已保存到: {scenario['output_dir']}")
            else:
                print(f"  警告：未找到结果文件 {results_file}")

        except Exception as e:
            print(f"  测试场景失败: {e}")
            import traceback

            traceback.print_exc()

    # 比较分析结果
    print(f"\n{'='*60}")
    print("分析结果比较")
    print(f"{'='*60}")

    if len(all_results) >= 2:
        print("钢板变形比较 (最大位移, mm):")
        print("-" * 50)
        for scenario_name, results in all_results.items():
            if "plate_analysis" in results:
                plate_analysis = results["plate_analysis"]
                deform_data = plate_analysis.get("plate_deformation_analysis", {})
                indicators = deform_data.get("performance_indicators", {})
                max_disp = indicators.get("max_plate_displacement_mm", 0)
                pressure = plate_analysis.get("analysis_parameters", {}).get("internal_pressure_pa", 0)
                print(f"  {scenario_name:25s}: {max_disp:8.4f} mm (压力: {pressure:4.0f} Pa)")

    # 生成分析报告
    report_file = "results/opensees_plate_analysis_report.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    report_data = {
        "test_summary": {
            "total_scenarios": len(test_scenarios),
            "successful_analyses": len(all_results),
            "model_info": {
                "nodes": len(model.points),
                "beams": len(model.beams),
                "plates": len(model.plates) if model.plates else 0,
            },
        },
        "scenario_results": all_results,
        "analysis_metadata": {
            "analysis_type": "pressurized_building_steel_plates",
            "element_type": "ShellMITC4",
            "material_model": "ElasticIsotropic",
            "units": "N_mm_s_tonne",
        },
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细分析报告已保存到: {report_file}")

    # 性能建议
    print(f"\n{'='*60}")
    print("结构性能评估与建议")
    print(f"{'='*60}")

    if all_results:
        # 分析结果并提供建议
        max_displacements = []
        for scenario_name, results in all_results.items():
            if "plate_analysis" in results:
                plate_analysis = results["plate_analysis"]
                deform_data = plate_analysis.get("plate_deformation_analysis", {})
                indicators = deform_data.get("performance_indicators", {})
                max_disp = indicators.get("max_plate_displacement_mm", 0)
                max_displacements.append(max_disp)

        if max_displacements:
            overall_max_disp = max(max_displacements)
            print(f"整体最大钢板位移: {overall_max_disp:.4f} mm")

            # 给出工程建议
            if overall_max_disp < 5.0:
                print("✓ 钢板变形在可接受范围内")
            elif overall_max_disp < 15.0:
                print("⚠ 钢板变形较大，建议检查设计")
                print("  - 考虑增加钢板厚度")
                print("  - 优化支撑结构")
            else:
                print("✗ 钢板变形过大，需要重新设计")
                print("  - 增加钢板厚度")
                print("  - 增加中间支撑")
                print("  - 降低内部压力")

    print(f"\n{'='*60}")
    print("OpenSees钢板建模测试完成")
    print(f"{'='*60}")

    return True


if __name__ == "__main__":
    try:
        success = test_opensees_plate_modeling()
        if success:
            print("\n测试成功完成！")
        else:
            print("\n测试失败。")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
