from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Union

import gym
import numpy as np

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.graph.structural_graph import BeamAsEdgeGraph, BeamAsNodeGraph
from src.structure.mesh_generator import MeshGenerator, MeshParameters
from src.structure.random_structure import RandomStructure
from src.structure.structural_model import (
    Response,
    SectionLibrary,
    StructuralModel,
    StructuralParameters,
    Window,
)
from src.utils.logger import Logger
from src.utils.model_utils import create_structural_model, load_config


class StructuralEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        initial_params_config: Union[str, dict],
        logger: Logger,
    ):
        super(StructuralEnv, self).__init__()
        self.logger = logger

        self.section_library = SectionLibrary()
        self.opensees_analyzer = OpenSeesAnalyzer()

        if isinstance(initial_params_config, str):
            initial_full_config = load_config(initial_params_config)
        else:
            initial_full_config = initial_params_config

        analysis_config = initial_full_config["analysis_params"]
        self.max_stress = analysis_config["max_stress"]
        self.max_disp = analysis_config["max_disp"]
        self.env_config = initial_full_config["env_config"]
        # base_params is a dictionary holding the initial configuration for the structure.
        # It's used as a template and is copied to current_params at the start of each episode.
        self.base_params: dict = initial_full_config["structural_params"]

        # current_params holds the mutable state of the environment's structural parameters during an episode.
        # It is modified by the agent's actions.
        self.current_params: Optional[dict] = None

        self.current_structural_model: Optional[StructuralModel] = None
        self.current_graph_converter: Optional[Union[BeamAsNodeGraph, BeamAsEdgeGraph]] = None
        self.current_pyg_data_state = None  # Will be a PyG Data object

        self._max_steps_per_episode = initial_full_config["rl_params"]["max_steps_per_episode"]
        self._current_step_count = 0

        self._last_max_disp_abs = 0.0
        self._last_max_stress = 0.0
        self._last_model_volume = 0.0
        self._last_action = None

    def _calculate_volume_for_params(self, params_dict: dict) -> float:
        """根据给定的参数字典计算并返回结构体积"""
        # 创建一个临时的配置对象
        temp_config = {"structural_params": params_dict}
        # 使用工具函数创建模型
        model = create_structural_model(temp_config)
        # 计算并返回体积
        return model.calculate_mass() / 7850.0

    def _calculate_volume_bounds(self):
        """计算当前结构参数下的最小和最大可能体积"""
        # 最小体积参数
        min_params = self.current_params.copy()
        min_params["frame_section_index"] = 0
        min_params["main_beam_section_index"] = 0
        min_params["secondary_beam_section_index"] = 0
        min_params["brace_section_index"] = 0
        self.min_volume = self._calculate_volume_for_params(min_params)

        # 最大体积参数
        max_params = self.current_params.copy()
        section_counts = self.section_library.section_counts
        max_params["frame_section_index"] = section_counts["方钢"] - 1
        max_params["main_beam_section_index"] = section_counts["H型钢-主梁"] - 1
        max_params["secondary_beam_section_index"] = section_counts["H型钢-次梁"] - 1
        max_params["brace_section_index"] = section_counts["C型钢"] - 1
        self.max_volume = self._calculate_volume_for_params(max_params)

    def _get_current_structural_parameters_obj(self) -> StructuralParameters:
        """
        Creates a StructuralParameters object from the current parameters dictionary.
        This involves reconstructing complex objects like Windows.
        """
        if self.current_params is None:
            raise ValueError("Current parameters are not set. Did you forget to call reset()?")

        # Start with a copy of the current parameter dictionary
        params_data_dict = self.current_params.copy()

        # Re-create Window objects from the config dictionary format
        windows_list = []
        if "windows" in params_data_dict and params_data_dict["windows"]:
            for w_config in params_data_dict["windows"]:
                window_params = {
                    "y_position": w_config.get("y_position", 0.0),
                    "width": params_data_dict["main_beam_spacing"],
                    "height": params_data_dict["win_height"],
                    "bottom_height": (
                        0.0 if w_config.get("is_door", False) else params_data_dict["win_bottom"]
                    ),
                    "on_face": w_config.get("on_face", "front"),
                    "is_door": w_config.get("is_door", False),
                }
                windows_list.append(Window(**window_params))
        params_data_dict["windows"] = windows_list

        return StructuralParameters(**params_data_dict)

    def _generate_model_and_state_from_current_params(self):
        """
        Generates the StructuralModel based on current parameters,
        creates its mesh, and converts it to a PyG Data object.
        Returns True if successful, False otherwise.
        """
        params_obj = self._get_current_structural_parameters_obj()
        self.current_structural_model = StructuralModel(params_obj, self.section_library)
        self.current_structural_model.generate_grid_mesh()

        try:
            max_indicators = self._run_fem_analysis()
            if not max_indicators:
                print("Warning (StructuralEnv): Failed to run FEM analysis.")
                return False, None
        except Exception as e:
            print(f"Error during OpenSees analysis in _generate_model_and_state_from_current_params: {e}")
            return False, None

        try:
            if self.env_config["graph_type"] == "beam_as_node":
                self.current_graph_converter = BeamAsNodeGraph(self.current_structural_model)
            elif self.env_config["graph_type"] == "beam_as_edge":
                self.current_graph_converter = BeamAsEdgeGraph(self.current_structural_model)
            else:
                raise ValueError(f"Invalid graph type: {self.env_config['graph_type']}")

            env_info = {
                "frame_length": self.current_params["length"],
                "pressure": self.current_params["pressure"],
            }
            self.current_pyg_data_state = self.current_graph_converter.to_pyg_data(env_info)
        except Exception as e:
            print(f"Error during graph conversion in _generate_model_and_state_from_current_params: {e}")
            return False, None

        return True, max_indicators

    def reset(self, options: Optional[dict] = None):
        # At the start of an episode, create a mutable copy of the base parameters.
        self.current_params = self.base_params.copy()

        # If a specific structural config is passed via options, use it.
        # Otherwise, use the base config from initialization.
        if options and options.get("random_structure", False):
            random_keys = options["random_structure"].get("random_keys", [])
            random_params = asdict(RandomStructure(self.section_library).generate_random_parameters())
            for key in random_keys:
                if key in self.current_params:
                    self.current_params[key] = random_params[key]

        # 添加随机起点扰动以提升探索连通性
        if options and options.get("random_start_perturbation", False):
            perturbation_prob = options.get("perturbation_prob", 0.3)
            if np.random.random() < perturbation_prob:
                # 随机扰动截面索引在小邻域内
                section_types = [
                    "frame_section_index",
                    "main_beam_section_index",
                    "secondary_beam_section_index",
                    "brace_section_index",
                ]
                frame_types = ["frame", "main", "secondary", "brace"]

                for section_type, frame_type in zip(section_types, frame_types):
                    max_idx = self.section_library.section_num_by_frame_type(frame_type) - 1
                    current_idx = self.current_params[section_type]

                    # 在当前值附近±2的范围内随机扰动
                    perturbation = np.random.randint(-2, 3)
                    new_idx = np.clip(current_idx + perturbation, 0, max_idx)
                    self.current_params[section_type] = new_idx

        self._current_step_count = 0

        generation_success, max_indicators = self._generate_model_and_state_from_current_params()
        if not generation_success:
            print("Critical Error (StructuralEnv): Failed to generate a valid model/state at reset.")
            raise ValueError("Failed to generate a valid model/state at reset.")

        if max_indicators:
            self._last_max_disp_abs = max_indicators["max_abs_displacement"]
            self._last_max_stress = max_indicators["max_abs_stress"]
        else:
            raise ValueError("Failed to run FEM analysis at reset.")

        self._last_model_volume = self._calculate_model_volume()
        self.best_known_volume = self._last_model_volume
        self._calculate_volume_bounds()
        self.normalized_volume = self._calculate_normalized_volume(self._last_model_volume)

        self.logger.info(
            "\n\n"
            + f"Start new episode: Length: {self.current_params['length']}, Pressure: {self.current_params['pressure']}, Volumn bounds: {self.min_volume:.2f} ~ {self.max_volume:.2f}, Current Volume: {self._last_model_volume:.2f}, Total Force: {self.total_force_kN:.2f} kN"
        )

        info = {
            "max_displacement_mm": self._last_max_disp_abs,
            "max_stress": self._last_max_stress,
            "model_volume_m3": self._last_model_volume,
            "frame_length": self.current_params["length"],
            "normalized_volume": self.normalized_volume,
        }
        return self.current_pyg_data_state, info

    # def _decode_action(self, action_vector: np.ndarray):
    #     """
    #     Decodes the continuous action vector from the agent (assumed to be in [0, 1])
    #     into actual structural parameter values.
    #     """
    #     param_map = self.env_config["parameter_action_mapping"]

    #     # Main Beam Spacing
    #     cfg = param_map["main_beam_spacing"]
    #     raw_val = cfg["bounds"][0] + (1 - action_vector[cfg["action_idx"]]) * (
    #         cfg["bounds"][1] - cfg["bounds"][0]
    #     )
    #     self.current_main_beam_spacing = np.clip(
    #         round(raw_val / cfg["module"]) * cfg["module"], cfg["bounds"][0], cfg["bounds"][1]
    #     )

    #     # Brace Spacing
    #     cfg = param_map["brace_spacing"]
    #     raw_val = cfg["bounds"][0] + action_vector[cfg["action_idx"]] * (cfg["bounds"][1] - cfg["bounds"][0])
    #     num_spans = np.clip(round(raw_val), cfg["bounds"][0], cfg["bounds"][1])
    #     if num_spans > 1:
    #         brace_spacing = (self.current_main_beam_spacing / num_spans).round(3)
    #     else:
    #         brace_spacing = (1.3 * (self.current_main_beam_spacing / 2)).round(3)
    #     self.current_brace_spacing = brace_spacing if brace_spacing > 0.3 else 0.3

    #     def _decode_single_action(section_index_name):
    #         cfg = param_map[section_index_name]
    #         num_options = cfg["num_options"]
    #         ## 1. interpret action as normalized index of the section
    #         raw_idx = action_vector[cfg["action_idx"]] * (num_options - 1)  # Scale [0,1] to [0, N-1]
    #         return np.clip(int(round(raw_idx)), 0, num_options - 1)

    #         # ### 2. interpret action as increase or decrease of the section index
    #         # current_section_index = getattr(self, f"current_{section_index_name}")
    #         # if action_vector[cfg["action_idx"]] > 0.5:
    #         #     updated_section_index = current_section_index + 1
    #         # else:
    #         #     updated_section_index = current_section_index - 1
    #         # return np.clip(updated_section_index, 0, num_options - 1)

    #     # 4. Frame Section Index
    #     self.current_frame_section_index = _decode_single_action("frame_section_index")

    #     # 5. Main Beam Section Index
    #     self.current_main_beam_section_index = _decode_single_action("main_beam_section_index")

    #     # 6. Secondary Beam Section Index
    #     self.current_secondary_beam_section_index = _decode_single_action("secondary_beam_section_index")

    #     # 7. Brace Section Index
    #     self.current_brace_section_index = _decode_single_action("brace_section_index")

    def _decode_action_for_size(self, action: np.ndarray):
        if self.env_config["graph_type"] == "beam_as_node":
            # beam_id = self.current_graph_converter.get_beam_id_from_pyg_idx(action)
            # beam_obj = self.current_structural_model.beams[beam_id]
            # frame_type = beam_obj.section
            # if frame_type.lower() in ["main", "secondary"]:
            #     current_sec_idx_name = f"{frame_type.lower()}_beam_section_index"
            # else:
            #     current_sec_idx_name = f"{frame_type.lower()}_section_index"
            # current_sec_idx = self.current_params[current_sec_idx_name]

            # section_idx_upper_bound = self.section_library.section_num_by_frame_type(frame_type.lower())
            # updated_sec_idx = np.clip(current_sec_idx + 1, 0, section_idx_upper_bound - 1)
            # self.current_params[current_sec_idx_name] = updated_sec_idx

            ## 0: 降低框架梁截面尺寸，1: 提高框架梁截面尺寸，2: 降低主梁截面尺寸，3: 提高主梁截面尺寸，4: 降低次梁截面尺寸，5: 提高次梁截面尺寸，6: 降低支撑梁截面尺寸，7: 提高支撑梁截面尺寸
            is_valid = True
            if action == 0:
                if self.current_params["frame_section_index"] == 0:
                    return False
                self.current_params["frame_section_index"] = np.clip(
                    self.current_params["frame_section_index"] - 1,
                    0,
                    self.section_library.section_num_by_frame_type("frame") - 1,
                )
            elif action == 1:
                if (
                    self.current_params["frame_section_index"]
                    == self.section_library.section_num_by_frame_type("frame") - 1
                ):
                    return False
                self.current_params["frame_section_index"] = np.clip(
                    self.current_params["frame_section_index"] + 1,
                    0,
                    self.section_library.section_num_by_frame_type("frame") - 1,
                )
            elif action == 2:
                if self.current_params["main_beam_section_index"] == 0:
                    return False
                self.current_params["main_beam_section_index"] = np.clip(
                    self.current_params["main_beam_section_index"] - 1,
                    0,
                    self.section_library.section_num_by_frame_type("main") - 1,
                )
            elif action == 3:
                if (
                    self.current_params["main_beam_section_index"]
                    == self.section_library.section_num_by_frame_type("main") - 1
                ):
                    return False
                self.current_params["main_beam_section_index"] = np.clip(
                    self.current_params["main_beam_section_index"] + 1,
                    0,
                    self.section_library.section_num_by_frame_type("main") - 1,
                )
            elif action == 4:
                if self.current_params["secondary_beam_section_index"] == 0:
                    return False
                self.current_params["secondary_beam_section_index"] = np.clip(
                    self.current_params["secondary_beam_section_index"] - 1,
                    0,
                    self.section_library.section_num_by_frame_type("secondary") - 1,
                )
            elif action == 5:
                if (
                    self.current_params["secondary_beam_section_index"]
                    == self.section_library.section_num_by_frame_type("secondary") - 1
                ):
                    return False
                self.current_params["secondary_beam_section_index"] = np.clip(
                    self.current_params["secondary_beam_section_index"] + 1,
                    0,
                    self.section_library.section_num_by_frame_type("secondary") - 1,
                )
            elif action == 6:
                if self.current_params["brace_section_index"] == 0:
                    return False
                self.current_params["brace_section_index"] = np.clip(
                    self.current_params["brace_section_index"] - 1,
                    0,
                    self.section_library.section_num_by_frame_type("brace") - 1,
                )
            elif action == 7:
                if (
                    self.current_params["brace_section_index"]
                    == self.section_library.section_num_by_frame_type("brace") - 1
                ):
                    return False
                self.current_params["brace_section_index"] = np.clip(
                    self.current_params["brace_section_index"] + 1,
                    0,
                    self.section_library.section_num_by_frame_type("brace") - 1,
                )
            return is_valid
        else:
            raise ValueError(f"Invalid graph type: {self.env_config['graph_type']}")

    def _decode_action_for_spacing(self, action: np.ndarray):
        # main_beam_spacing_range = np.linspace(1.8, 0.3, 16)
        # brace_spacing_range = np.linspace(1.0, 0.3, 10)
        if self.env_config["graph_type"] == "beam_as_node":
            # beam_id = self.current_graph_converter.get_beam_id_from_pyg_idx(action)
            # beam_obj = self.current_structural_model.beams[beam_id]
            # frame_type = beam_obj.section
            # if frame_type.lower() in ["main", "secondary"]:
            #     current_beam_spacing_name = f"main_beam_spacing"
            #     current_beam_spacing = self.current_params[current_beam_spacing_name]
            #     updated_beam_spacing = np.clip(current_beam_spacing - 0.1, 0.3, 1.8)
            # elif frame_type.lower() == "brace":
            #     current_beam_spacing_name = f"brace_spacing"
            #     current_beam_spacing = self.current_params[current_beam_spacing_name]
            #     updated_beam_spacing = np.clip(current_beam_spacing - 0.1, 0.3, 1.0)
            # else:
            #     return None
            # self.current_params[current_beam_spacing_name] = updated_beam_spacing

            ### 0: 降低主梁间距，1: 提高主梁间距，2: 降低支撑梁间距，3: 提高支撑梁间距
            if action == 0:
                self.current_params["main_beam_spacing"] = np.clip(
                    self.current_params["main_beam_spacing"] - 0.05, 0.3, 1.8
                )
            elif action == 1:
                self.current_params["main_beam_spacing"] = np.clip(
                    self.current_params["main_beam_spacing"] + 0.05, 0.3, 1.8
                )
            elif action == 2:
                self.current_params["brace_spacing"] = np.clip(
                    self.current_params["brace_spacing"] - 0.05, 0.3, 1.0
                )
            elif action == 3:
                self.current_params["brace_spacing"] = np.clip(
                    self.current_params["brace_spacing"] + 0.05, 0.3, 1.0
                )
            return True
        else:
            raise ValueError(f"Invalid graph type: {self.env_config['graph_type']}")

    def _run_fem_analysis(self) -> Optional[Dict[str, float]]:
        # create mesh
        mesh_params = MeshParameters(max_grid_size=0.4)
        mesh_generator = MeshGenerator(self.current_structural_model, mesh_params)
        mesh_generator.generate_mesh()
        meshed_model = mesh_generator.meshed_model

        self.opensees_analyzer.setup_opensees_model(mesh_generator.meshed_model)
        self.opensees_analyzer.apply_gravity_and_pressure_loads(self.current_params["pressure"] * 1e3)
        self.total_force_kN = self.opensees_analyzer.total_equi_force_N / 1e3
        static_success = self.opensees_analyzer.run_static_analysis()
        if not static_success:
            return None

        # 填充梁端节点荷载
        nodal_loads = self.current_structural_model.calc_node_load(
            self.current_params["pressure"] * 1e3
        )  # 单位N
        for beam_id, beam_obj in self.current_structural_model.beams.items():
            beam_obj.start_node_load = nodal_loads[beam_obj.start_node] / 1e3  # 单位kN
            beam_obj.end_node_load = nodal_loads[beam_obj.end_node] / 1e3  # 单位kN

        # 填充梁的响应
        for beam_id, beam_obj in self.current_structural_model.beams.items():
            meshed_beams_id = mesh_generator.original_to_meshed_beams[beam_id]
            if len(meshed_beams_id) == 0:
                continue
            beam_obj.response = Response(max_disp_abs=0.0, max_stress=0.0)
            stresses = self.opensees_analyzer.get_beam_stresses(meshed_beams_id)
            stresses_arr = np.array(list(stresses.values()))
            beam_obj.response.max_stress = np.max(np.abs(stresses_arr))

            # meshed beam nodes
            meshed_beam_nodes = []
            for meshed_beam_id in meshed_beams_id:
                meshed_beam_nodes.append(meshed_model.beams[meshed_beam_id].start_node)
                meshed_beam_nodes.append(meshed_model.beams[meshed_beam_id].end_node)
            displacements = self.opensees_analyzer.get_node_displacements(meshed_beam_nodes)
            displacements_arr = np.array(list(displacements.values()))[:, :3]
            beam_obj.response.max_disp_abs = np.max(np.abs(displacements_arr))

        max_indicators = self.opensees_analyzer.get_max_indicators()
        return max_indicators

    def _calculate_model_volume(self) -> float:
        return self.current_structural_model.calculate_mass() / 7850.0

    # def _calculate_reward(self, max_disp_abs: float, max_stress: float, model_volume: float):
    #     disp_constraint = 14.4
    #     stress_constraint = 355.0
    #     max_volume = 1.615

    #     # 1. 约束违反惩罚项（指数增长确保强惩罚）
    #     disp_violation = max(0, (max_disp_abs - disp_constraint) / disp_constraint)
    #     stress_violation = max(0, (max_stress - stress_constraint) / stress_constraint)
    #     penalty = (disp_violation * 1) + (stress_violation * 1.5)  # 权重可调

    #     # 2. 基础奖励：体积的负值（体积越小奖励越高）
    #     if disp_violation == 0 and stress_violation == 0:
    #         base_reward = 1 - model_volume / max_volume
    #     else:
    #         base_reward = 0

    #     # 3. 总奖励 = 基础奖励 - 惩罚项
    #     reward = base_reward - penalty

    #     self.logger.info(
    #         f"Reward: {reward:.2f}, Penalty: {-penalty:.2f}, Disp: {disp_violation*1:.2f}, Stress: {stress_violation*1.5:.2f}, Base Reward: {base_reward:.2f}"
    #     )

    #     return reward

    # def _calculate_reward(self, max_disp_abs: float, max_stress: float, model_volume: float):
    #     disp_constraint = self.max_disp
    #     stress_constraint = self.max_stress

    #     if max_disp_abs < disp_constraint:
    #         delta_c1 = 1.0
    #     else:
    #         delta_c1 = (
    #             (self._last_max_disp_abs - max_disp_abs)
    #             / disp_constraint
    #             / (max_disp_abs / disp_constraint + 1)
    #         )

    #     if max_stress < stress_constraint:
    #         delta_c2 = 1.0
    #     else:
    #         delta_c2 = (
    #             (self._last_max_stress - max_stress)
    #             / stress_constraint
    #             / (max_stress / stress_constraint + 1)
    #         )

    #     delta_v = model_volume - self._last_model_volume

    #     if delta_c1 == 1 and delta_c2 == 1:
    #         reward = -delta_v * 100 if delta_v < 0 else -delta_v * 200
    #     else:
    #         reward = -(2 - delta_c1 - delta_c2) * 10

    #     # self.logger.info(
    #     #     f"Reward: {reward:.2f}, delta_c1: {delta_c1:.2f}, delta_c2: {delta_c2:.2f}, delta_v: {delta_v:.2f}"
    #     # )

    #     return reward

    # def _calculate_reward(self, max_disp_abs, max_stress, model_volume):
    #     disp_constraint = self.max_disp
    #     stress_constraint = self.max_stress

    #     # 1. 计算约束满足度（0-1连续值）
    #     # disp_satisfaction = max(0, 1 - max_disp_abs / disp_constraint)
    #     # stress_satisfaction = max(0, 1 - max_stress / stress_constraint)
    #     # constraint_satisfaction = (disp_satisfaction + stress_satisfaction) / 2

    #     # 2. 材料效率奖励（鼓励用更少材料满足约束）
    #     # 基准材料用量（经验值或初始设计）
    #     baseline_volume = 1.73
    #     volume_efficiency = max(0.1, baseline_volume / model_volume)  # 当前用量/基准用量

    #     # 3. 改进奖励（与上一步比较）
    #     delta_disp = self._last_max_disp_abs - max_disp_abs
    #     delta_stress = self._last_max_stress - max_stress
    #     delta_volume = self._last_model_volume - model_volume

    #     # 4. 核心奖励计算
    #     if max_disp_abs < disp_constraint and max_stress < stress_constraint:
    #         # 满足约束时的奖励 = 基础奖励 + 材料效率奖励
    #         reward = 20.0 + 10 * volume_efficiency
    #         self.logger.info(f"Reward: {reward:.2f}, volume_efficiency: {volume_efficiency:.2f}")
    #     else:
    #         # 未满足时的奖励 = 约束改进奖励 + 材料节省奖励
    #         improvement_reward = 30 * delta_disp / disp_constraint + delta_stress / stress_constraint
    #         volume_penalty = 5 * delta_volume / baseline_volume
    #         if volume_penalty == 0:
    #             volume_penalty = -2

    #         reward = improvement_reward + volume_penalty

    #         self.logger.info(
    #             f"Reward: {reward:.2f}, improvement_reward: {improvement_reward:.2f}, volume_penalty: {volume_penalty:.2f}"
    #         )

    #     return reward

    def _calculate_reward_for_size(self, max_disp_abs, max_stress, model_volume):
        """
        改进的两阶段奖励函数:
        1. 不可行阶段: 只奖励约束改善，不考虑体积
        2. 可行阶段: 主要奖励体积最小化
        """
        disp_constraint = self.max_disp
        stress_constraint = self.max_stress

        # 计算约束违反度
        disp_ratio = max_disp_abs / disp_constraint
        stress_ratio = max_stress / stress_constraint
        violation = max(0, disp_ratio - 1) + max(0, stress_ratio - 1)

        # 计算上一步的约束违反度
        last_disp_ratio = self._last_max_disp_abs / disp_constraint
        last_stress_ratio = self._last_max_stress / stress_constraint
        last_violation = max(0, last_disp_ratio - 1) + max(0, last_stress_ratio - 1)

        # 约束改善量
        delta_violation = last_violation - violation

        # 体积相关
        normalized_volume = self._calculate_normalized_volume(model_volume)
        delta_volume = self._last_model_volume - model_volume

        # 步数惩罚(鼓励更快收敛)
        step_penalty = 0.1

        # 两阶段奖励设计
        if violation > 0:
            # 不可行阶段: 只关注约束改善
            c_v = 8.0  # 约束改善系数
            w_v = 15.0  # 约束违反惩罚权重

            shaping_reward = c_v * delta_violation
            base_penalty = -w_v * violation
            reward = shaping_reward + base_penalty - step_penalty

            self.logger.info(
                f"[不可行] r: {reward:.2f}, shaping: {shaping_reward:.2f}, base: {base_penalty:.2f}, "
                f"δviolation: {delta_violation:.3f}, violation: {violation:.3f}"
            )
        else:
            # 可行阶段: 主要奖励体积最小化
            R_feasible = 5.0
            alpha = 40.0  # 体积优化权重

            # 基础可行奖励 + 体积优化奖励
            ongoing_reward = R_feasible + alpha * (1 - normalized_volume) - step_penalty

            # 如果刚从不可行变为可行，给额外奖励
            terminal_bonus = 0
            if last_violation > 0:  # 刚刚达到可行
                B_success = 80.0
                terminal_bonus = B_success * (1 - normalized_volume)

            reward = ongoing_reward + terminal_bonus

            self.logger.info(
                f"[可行] r: {reward:.2f}, ongoing: {ongoing_reward:.2f}, bonus: {terminal_bonus:.2f}, "
                f"norm_vol: {normalized_volume:.3f}, vol: {model_volume:.3f}"
            )

        return reward

    def get_action_mask_for_size(self):
        """
        获取当前状态下的动作掩码
        返回长度为8的布尔数组，True表示该动作有效
        """
        mask = [True] * 8

        # 检查框架梁 (actions 0, 1)
        if self.current_params["frame_section_index"] == 0:
            mask[0] = False  # 不能再降低
        if (
            self.current_params["frame_section_index"]
            >= self.section_library.section_num_by_frame_type("frame") - 1
        ):
            mask[1] = False  # 不能再提高

        # 检查主梁 (actions 2, 3)
        if self.current_params["main_beam_section_index"] == 0:
            mask[2] = False
        if (
            self.current_params["main_beam_section_index"]
            >= self.section_library.section_num_by_frame_type("main") - 1
        ):
            mask[3] = False

        # 检查次梁 (actions 4, 5)
        if self.current_params["secondary_beam_section_index"] == 0:
            mask[4] = False
        if (
            self.current_params["secondary_beam_section_index"]
            >= self.section_library.section_num_by_frame_type("secondary") - 1
        ):
            mask[5] = False

        # 检查支撑梁 (actions 6, 7)
        if self.current_params["brace_section_index"] == 0:
            mask[6] = False
        if (
            self.current_params["brace_section_index"]
            >= self.section_library.section_num_by_frame_type("brace") - 1
        ):
            mask[7] = False

        return mask

    def get_action_mask_for_spacing(self):
        """
        获取当前状态下的动作掩码
        返回长度为8的布尔数组，True表示该动作有效
        """
        mask = [True] * 4
        if self.current_params["main_beam_spacing"] <= 0.3:
            mask[0] = False
        if self.current_params["main_beam_spacing"] >= 1.8:
            mask[1] = False
        if self.current_params["brace_spacing"] <= 0.3:
            mask[2] = False
        if self.current_params["brace_spacing"] >= 1.0:
            mask[3] = False
        return mask

    def get_action_mask_for_both(self):
        mask = [True] * 12
        size_mask = self.get_action_mask_for_size()
        spacing_mask = self.get_action_mask_for_spacing()
        mask[0:8] = size_mask
        mask[8:12] = spacing_mask
        return mask

    def _decode_action_for_size_with_mask(self, action: int):
        """
        带动作掩码的动作解码，无效动作作为no-op处理
        """
        mask = self.get_action_mask_for_size()

        if not mask[action]:
            # 无效动作，作为no-op处理
            return True, True  # (is_valid, is_noop)

        # 有效动作，按原逻辑执行
        if action == 0:
            self.current_params["frame_section_index"] -= 1
        elif action == 1:
            self.current_params["frame_section_index"] += 1
        elif action == 2:
            self.current_params["main_beam_section_index"] -= 1
        elif action == 3:
            self.current_params["main_beam_section_index"] += 1
        elif action == 4:
            self.current_params["secondary_beam_section_index"] -= 1
        elif action == 5:
            self.current_params["secondary_beam_section_index"] += 1
        elif action == 6:
            self.current_params["brace_section_index"] -= 1
        elif action == 7:
            self.current_params["brace_section_index"] += 1

        return True, False  # (is_valid, is_noop)

    def _decode_action_for_spacing_with_mask(self, action: int):
        mask = self.get_action_mask_for_spacing()
        if not mask[action]:
            return True, True  # (is_valid, is_noop)
        if action == 0:
            self.current_params["main_beam_spacing"] -= 0.05
        elif action == 1:
            self.current_params["main_beam_spacing"] += 0.05
        elif action == 2:
            self.current_params["brace_spacing"] -= 0.05
        elif action == 3:
            self.current_params["brace_spacing"] += 0.05
        return True, False  # (is_valid, is_noop)

    def _decode_action_for_both_with_mask(self, action: int):
        mask = self.get_action_mask_for_both()
        if not mask[action]:
            return True, True  # (is_valid, is_noop)
        if action < 8:
            self._decode_action_for_size_with_mask(action)
        else:
            self._decode_action_for_spacing_with_mask(action - 8)
        return True, False  # (is_valid, is_noop)

    def _calculate_reward_for_spacing(self, max_disp_abs, max_stress, model_volume):
        return self._calculate_reward_for_size(max_disp_abs, max_stress, model_volume)
        # disp_constraint = self.max_disp
        # stress_constraint = self.max_stress
        # normalized_volume = self._calculate_normalized_volume(model_volume)

        # # 1. 计算关键指标
        # disp_ratio = max_disp_abs / disp_constraint
        # stress_ratio = max_stress / stress_constraint

        # # 2. 约束满足度（0-1）
        # feasibility = (disp_ratio + stress_ratio) / 2 if (disp_ratio < 1 and stress_ratio < 1) else 0.0

        # # 3. 材料效率奖励（核心）
        # # 鼓励接近最优材料用量
        # volume_efficiency = 1 - normalized_volume**2

        # # 4. 改进潜力指标
        # improvement_potential = 0
        # if feasibility > 0.5:
        #     # 计算当前解与理论最优的材料差距
        #     current_to_optimal = max(0, (model_volume - self.best_known_volume) / self.best_known_volume)
        #     improvement_potential = 1 / (current_to_optimal + 0.1)

        # # 5. 核心奖励计算
        # if feasibility > 0:
        #     reward = (
        #         10 * feasibility  # 基础可行奖励
        #         + 30 * volume_efficiency  # 材料效率（主驱动）
        #         + 5 * improvement_potential  # 引导寻找更优解
        #         - 0.5 * (disp_ratio + stress_ratio)  # 持续优化约束
        #     )
        # else:
        #     reward = -2

        # # 6. 发现新最优解的额外奖励
        # if feasibility > 0.8 and model_volume < self.best_known_volume:
        #     reward += 50  # 大幅奖励突破性改进
        #     self.best_known_volume = model_volume  # 更新已知最优

        # self.logger.info(
        #     f"Reward: {reward:.2f}, feasibility: {feasibility:.2f}, volume_efficiency: {volume_efficiency:.2f}, improvement_potential: {improvement_potential:.2f}, disp_ratio: {disp_ratio:.2f}, stress_ratio: {stress_ratio:.2f}, model_volume: {model_volume:.2f}, best_known_volume: {self.best_known_volume:.2f}"
        # )

        # return reward

    def _calculate_reward_for_both(self, max_disp_abs, max_stress, model_volume):
        """
        改进的多任务奖励函数（同时优化尺寸和间距）:
        1. 向量化约束违反度，优先处理最严重违反
        2. 不平衡惩罚，避免对某一约束过度调整
        3. 动作成本区分，间距调整成本更高
        4. 可行后贴边减材策略
        """
        disp_constraint = self.max_disp
        stress_constraint = self.max_stress

        # 计算向量化约束违反度
        v_disp = max(0, max_disp_abs / disp_constraint - 1)
        v_stress = max(0, max_stress / stress_constraint - 1)

        # 计算上一步的约束违反度
        last_v_disp = max(0, self._last_max_disp_abs / disp_constraint - 1)
        last_v_stress = max(0, self._last_max_stress / stress_constraint - 1)

        # 距离指标
        d_l2 = np.sqrt(v_disp**2 + v_stress**2)
        d_inf = max(v_disp, v_stress)
        last_d_l2 = np.sqrt(last_v_disp**2 + last_v_stress**2)
        last_d_inf = max(last_v_disp, last_v_stress)

        # 距离改善量
        delta_d_l2 = last_d_l2 - d_l2
        delta_d_inf = last_d_inf - d_inf

        # 各维约束改善量
        delta_v_disp = last_v_disp - v_disp
        delta_v_stress = last_v_stress - v_stress

        # 不平衡指标
        imbalance = abs(v_disp - v_stress)

        # 体积相关
        normalized_volume = self._calculate_normalized_volume(model_volume)

        # 步数惩罚
        step_penalty = 0.1

        # 获取最后一个动作（用于动作成本）
        action_cost = self._get_action_cost()

        # 两阶段奖励设计
        if v_disp > 0 or v_stress > 0:  # 不可行阶段
            # 超参数
            c_delta = 8.0  # 距离改善奖励系数
            c2 = 2.0  # L2距离改善系数
            w_d = 15.0  # 距离惩罚权重
            beta = 2.0  # 不平衡惩罚权重
            kappa_reg = 6.0  # 退化惩罚权重

            # 主要奖励：优先改善最严重违反（L∞为主，L2为辅）
            distance_reward = c_delta * delta_d_inf + c2 * delta_d_l2

            # 当前状态惩罚
            distance_penalty = -w_d * d_inf

            # 不平衡惩罚（避免单边过度优化）
            imbalance_penalty = -beta * imbalance

            # 退化惩罚（任何一维恶化都要惩罚）
            degradation_penalty = -kappa_reg * (max(-delta_v_disp, 0) + max(-delta_v_stress, 0))

            # 总奖励
            reward = (
                distance_reward
                + distance_penalty
                + imbalance_penalty
                + degradation_penalty
                - action_cost
                - step_penalty
            )

            self.logger.info(
                f"[不可行] r: {reward:.2f}, dist_r: {distance_reward:.2f}, dist_p: {distance_penalty:.2f}, "
                f"imbal_p: {imbalance_penalty:.2f}, degrad_p: {degradation_penalty:.2f}, "
                f"action_c: {action_cost:.2f}, v_disp: {v_disp:.3f}, v_stress: {v_stress:.3f}, "
                f"d_inf: {d_inf:.3f}, imbalance: {imbalance:.3f}"
            )

        else:  # 可行阶段
            # 超参数
            R0 = 5.0  # 基础可行奖励
            alpha = 50.0  # 体积优化权重
            mu = 8.0  # 过度安全惩罚权重
            B_success = 100.0  # 突破奖励系数
            tau = 0.98  # 贴边阈值

            # 基础可行奖励 + 体积优化
            base_reward = R0 + alpha * (1 - normalized_volume)

            # 计算当前约束比率
            disp_ratio = max_disp_abs / disp_constraint
            stress_ratio = max_stress / stress_constraint
            current_max_ratio = max(disp_ratio, stress_ratio)

            # 过度安全惩罚（鼓励贴边工作）
            distance_inside = max(0, tau - current_max_ratio)
            safety_penalty = -mu * distance_inside

            # 刚达可行的突破奖励
            breakthrough_bonus = 0
            if last_v_disp > 0 or last_v_stress > 0:  # 刚从不可行变为可行
                breakthrough_bonus = B_success * (1 - normalized_volume)

            reward = base_reward + safety_penalty + breakthrough_bonus - action_cost - step_penalty

            self.logger.info(
                f"[可行] r: {reward:.2f}, base: {base_reward:.2f}, safety_p: {safety_penalty:.2f}, "
                f"breakthrough: {breakthrough_bonus:.2f}, action_c: {action_cost:.2f}, "
                f"norm_vol: {normalized_volume:.3f}, max_ratio: {current_max_ratio:.3f}, "
                f"distance_inside: {distance_inside:.3f}"
            )

        # 回退保护：如果非主要违反维度恶化超过阈值
        epsilon = 0.05  # 5%的恶化阈值
        gamma = 8.0  # 回退惩罚权重

        # 识别主要违反维度
        if last_v_disp > last_v_stress:  # 位移是主要问题
            if delta_v_stress < -epsilon:  # 应力恶化超过阈值
                excess = abs(delta_v_stress) - epsilon
                reward -= gamma * excess
                self.logger.info(f"[回退保护] 应力恶化 {delta_v_stress:.3f}, 扣分: {gamma * excess:.2f}")
        elif last_v_stress > last_v_disp:  # 应力是主要问题
            if delta_v_disp < -epsilon:  # 位移恶化超过阈值
                excess = abs(delta_v_disp) - epsilon
                reward -= gamma * excess
                self.logger.info(f"[回退保护] 位移恶化 {delta_v_disp:.3f}, 扣分: {gamma * excess:.2f}")

        return reward

    def _get_action_cost(self):
        """
        计算动作成本，间距调整比尺寸调整成本更高
        """
        # 如果没有记录最后动作，返回0
        if not hasattr(self, "_last_action"):
            return 0.0

        action = self._last_action
        lambda_size = 0.1  # 尺寸调整成本
        lambda_space = 0.3  # 间距调整成本（更高）

        if action < 8:  # 尺寸调整动作 (0-7)
            return lambda_size
        else:  # 间距调整动作 (8-11)
            return lambda_space

    def _calculate_normalized_volume(self, model_volume: float) -> float:
        """计算归一化的材料用量"""
        return (model_volume - self.min_volume) / (self.max_volume - self.min_volume)

    def _calculate_volume_efficiency(self, model_volume: float) -> float:
        return self.total_force_kN / model_volume * 1e-3  # kN/dm³ 量级大约为3-5

    def step(self, action: int):
        # 记录动作用于计算动作成本
        self._last_action = action

        if self.env_config["opt_type"] == "size":
            is_valid, is_noop = self._decode_action_for_size_with_mask(action)
        elif self.env_config["opt_type"] == "spacing":
            is_valid, is_noop = self._decode_action_for_spacing_with_mask(action)
        elif self.env_config["opt_type"] == "both":
            is_valid, is_noop = self._decode_action_for_both_with_mask(action)
        else:
            raise ValueError(f"Invalid opt type: {self.env_config['opt_type']}")

        self._current_step_count += 1

        # 处理无效动作或no-op动作
        if not is_valid:
            # 对于spacing类型的旧式无效动作处理
            return (
                self.current_pyg_data_state,
                -0.5,  # 减小惩罚避免优势方差爆炸
                False,
                True if self._current_step_count >= self._max_steps_per_episode else False,
                {
                    "max_displacement_mm": self._last_max_disp_abs,
                    "max_stress": self._last_max_stress,
                    "model_volume_m3": self._last_model_volume,
                    "frame_length": self.current_params["length"],
                    "normalized_volume": self._calculate_normalized_volume(self._last_model_volume),
                    "volume_efficiency": self._calculate_volume_efficiency(self._last_model_volume),
                },
            )

        if is_noop:
            # no-op动作：给小惩罚但不改变状态
            return (
                self.current_pyg_data_state,
                -0.5,  # 小惩罚引导避免无效动作
                False,
                True if self._current_step_count >= self._max_steps_per_episode else False,
                {
                    "max_displacement_mm": self._last_max_disp_abs,
                    "max_stress": self._last_max_stress,
                    "model_volume_m3": self._last_model_volume,
                    "frame_length": self.current_params["length"],
                    "normalized_volume": self._calculate_normalized_volume(self._last_model_volume),
                    "volume_efficiency": self._calculate_volume_efficiency(self._last_model_volume),
                    "action_type": "noop",
                },
            )

        generation_success, max_indicators = self._generate_model_and_state_from_current_params()

        next_state = self.current_pyg_data_state
        reward = 0.0
        terminated = False  # Gym > 0.26 uses 'terminated' and 'truncated'
        truncated = False
        info = {}

        if (
            not generation_success
            or self.current_structural_model is None
            or not self.current_structural_model.beams
        ):
            reward = -10
            terminated = True
            print("Failed to generate valid structural model or PyG data from action.")
            info["error"] = "Failed to generate valid structural model or PyG data from action."
            return next_state, reward, terminated, truncated, info

        current_model_volume = self._calculate_model_volume()
        try:

            if max_indicators:
                max_disp_abs = max_indicators["max_abs_displacement"]
                max_stress = max_indicators["max_abs_stress"]
                if self.env_config["opt_type"] == "size":
                    reward = self._calculate_reward_for_size(max_disp_abs, max_stress, current_model_volume)
                elif self.env_config["opt_type"] == "spacing":
                    reward = self._calculate_reward_for_spacing(
                        max_disp_abs, max_stress, current_model_volume
                    )
                elif self.env_config["opt_type"] == "both":
                    reward = self._calculate_reward_for_both(max_disp_abs, max_stress, current_model_volume)
                else:
                    raise ValueError(f"Invalid opt type: {self.env_config['opt_type']}")

                self._last_max_disp_abs = max_disp_abs
                self._last_max_stress = max_stress
                self._last_model_volume = current_model_volume

                info["max_displacement_mm"] = max_disp_abs
                info["max_stress"] = max_stress
                info["model_volume_m3"] = current_model_volume
                info["frame_length"] = self.current_params["length"]
                info["normalized_volume"] = self._calculate_normalized_volume(current_model_volume)
                info["volume_efficiency"] = self._calculate_volume_efficiency(current_model_volume)

                if max_disp_abs < self.max_disp and max_stress < self.max_stress:
                    terminated = True

            else:
                print("OpenSees static analysis failed.")
                reward = -10
                terminated = True
                info["error"] = "OpenSees static analysis failed."

        except Exception as e:
            print(f"Error during env.step: {e}")
            # import traceback; traceback.print_exc() # For detailed debugging
            reward = -15.0  # 减小极端惩罚，避免优势方差爆炸
            terminated = True
            info["error"] = f"OpenSees analysis exception: {type(e).__name__} - {str(e)}"

        if self._current_step_count >= self._max_steps_per_episode:
            truncated = True  # Episode ends due to step limit

        return next_state, reward, terminated, truncated, info

    def render(self):
        # Optional: Implement visualization (e.g., using matplotlib or a 3D viewer)
        # For now, print current parameters.
        p = self.current_params
        print(f"Step: {self._current_step_count}")
        print("Frame Length: ", p["length"], "Pressure: ", p["pressure"])
        print(
            f"  Main Spacing: {p['main_beam_spacing']:.2f}, Secondary Spacing: {p['main_beam_spacing']:.2f}, Brace Spacing: {p['brace_spacing']:.2f}"
        )
        print(
            f"  Frame Idx: {p['frame_section_index']}, Main Idx: {p['main_beam_section_index']}, Secondary Idx: {p['secondary_beam_section_index']}, Brace Idx: {p['brace_section_index']}"
        )
        print(
            f"  Max Disp: {self._last_max_disp_abs:.2f}, Max Stress: {self._last_max_stress:.2f}, Model Volume: {self._last_model_volume:.2f}"
        )
        # print section info
        print(
            "Frame Section:",
            self.section_library.sections["方钢"][p["frame_section_index"]].name,
        )
        print(
            "Main Beam Section:",
            self.section_library.sections["H型钢-主梁"][p["main_beam_section_index"]].name,
        )
        print(
            "Secondary Beam Section:",
            self.section_library.sections["H型钢-次梁"][p["secondary_beam_section_index"]].name,
        )
        print(
            "Brace Section:",
            self.section_library.sections["C型钢"][p["brace_section_index"]].name,
        )

    def close(self):
        # Clean up any resources if needed (e.g., OpenSees model if ops.wipe() isn't sufficient)
        print("StructuralEnv closed.")


if __name__ == "__main__":
    env = StructuralEnv("config.yaml")
    env.reset()
    for i in range(1):
        max_beam_load = -float("inf")
        for beam_id, beam_obj in env.current_structural_model.beams.items():
            if np.max(np.abs(beam_obj.start_node_load)) > max_beam_load:
                max_beam_load = np.max(np.abs(beam_obj.start_node_load))
            if np.max(np.abs(beam_obj.end_node_load)) > max_beam_load:
                max_beam_load = np.max(np.abs(beam_obj.end_node_load))
        print(f"Max Beam Load: {max_beam_load:.2f}")
        env.reset()
        env.render()
        print("-" * 100)
