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
from src.utils.model_utils import load_config


class StructuralEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        initial_params_config: Union[str, dict],
    ):
        super(StructuralEnv, self).__init__()
        ts = datetime.now().strftime("%m%d_%H%M")
        self.logger = Logger(f"rewards_{ts}.log")

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
        self.base_volume = self._calculate_base_volume()
        self.logger.info(
            "\n\n"
            + f"Start new episode: Length: {self.current_params['length']}, Pressure: {self.current_params['pressure']}, Base Volume: {self.base_volume:.2f}, Current Volume: {self._last_model_volume:.2f}"
        )

        info = {
            "max_displacement_mm": self._last_max_disp_abs,
            "max_stress": self._last_max_stress,
            "model_volume_m3": self._last_model_volume,
            "frame_length": self.current_params["length"],
            "base_volume": self.base_volume,
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

    def _decode_action(self, action: np.ndarray):
        if self.env_config["graph_type"] == "beam_as_node":
            beam_id = self.current_graph_converter.get_beam_id_from_pyg_idx(action)
            beam_obj = self.current_structural_model.beams[beam_id]
            frame_type = beam_obj.section
            if frame_type.lower() in ["main", "secondary"]:
                current_sec_idx_name = f"{frame_type.lower()}_beam_section_index"
            else:
                current_sec_idx_name = f"{frame_type.lower()}_section_index"
            current_sec_idx = self.current_params[current_sec_idx_name]

            section_idx_upper_bound = self.section_library.section_num_by_frame_type(frame_type.lower())
            updated_sec_idx = np.clip(current_sec_idx + 1, 0, section_idx_upper_bound - 1)
            self.current_params[current_sec_idx_name] = updated_sec_idx
        else:
            raise ValueError(f"Invalid graph type: {self.env_config['graph_type']}")
        return beam_id

    def _run_fem_analysis(self) -> Optional[Dict[str, float]]:
        # create mesh
        mesh_params = MeshParameters(max_grid_size=0.4)
        mesh_generator = MeshGenerator(self.current_structural_model, mesh_params)
        mesh_generator.generate_mesh()
        meshed_model = mesh_generator.meshed_model

        self.opensees_analyzer.setup_opensees_model(mesh_generator.meshed_model)
        self.opensees_analyzer.apply_gravity_and_pressure_loads(self.current_params["pressure"] * 1e3)
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

    def _calculate_reward(self, max_disp_abs, max_stress, model_volume):
        disp_constraint = self.max_disp
        stress_constraint = self.max_stress

        # 1. 计算标准化的材料效率
        volume_ratio = model_volume / self.base_volume

        # 2. 计算约束满足度
        disp_satisfaction = max(0, 1 - max_disp_abs / disp_constraint)
        stress_satisfaction = max(0, 1 - max_stress / stress_constraint)
        constraint_satisfaction = (disp_satisfaction + stress_satisfaction) / 2

        # 3. 计算安全裕度（鼓励接近边界）
        disp_margin = (disp_constraint - max_disp_abs) / disp_constraint
        stress_margin = (stress_constraint - max_stress) / stress_constraint
        total_margin = disp_margin + stress_margin

        # 4. 核心奖励计算
        if max_disp_abs < disp_constraint and max_stress < stress_constraint:
            # 基础奖励
            base_reward = 10.0

            # 材料效率奖励（二次函数鼓励接近最优）
            efficiency_bonus = max(0, 10 * (1 - (volume_ratio - 0.9) ** 2))

            # 安全裕度奖励（鼓励接近边界）
            margin_bonus = max(0, 20 * (1 - total_margin))

            reward = base_reward + efficiency_bonus + margin_bonus
            self.logger.info(
                f"Reward: {reward:.2f}, efficiency_bonus: {efficiency_bonus:.2f}, margin_bonus: {margin_bonus:.2f}, disp: {max_disp_abs:.2f}, stress: {max_stress:.2f}"
            )
        else:
            # 改进奖励
            disp_improvement = max(0, (self._last_max_disp_abs - max_disp_abs) / disp_constraint)
            stress_improvement = max(0, (self._last_max_stress - max_stress) / stress_constraint)
            improvement_reward = 30 * disp_improvement + stress_improvement

            # 材料变化惩罚
            volume_change = model_volume - self._last_model_volume
            if volume_change < -0.01:  # 显著减少
                volume_penalty = volume_change * 40
            elif volume_change == 0:
                volume_penalty = -2
            elif volume_change < 0.02:  # 小幅变化
                volume_penalty = volume_change * (-10)
            else:  # 大幅增加
                volume_penalty = volume_change * (-50)

            reward = improvement_reward + volume_penalty + 5 * constraint_satisfaction
            self.logger.info(
                f"Reward: {reward:.2f}, improvement_reward: {improvement_reward:.2f}, volume_penalty: {volume_penalty:.2f}, constraint_satisfaction: {constraint_satisfaction:.2f}"
            )

        return reward

    def _calculate_base_volume(self):
        """计算基于结构尺寸的基准材料用量"""
        # 基于结构长度、跨度和荷载计算基准材料
        length_factor = self.current_params["length"]
        load_factor = self.current_params["pressure"] / 30  # 30是基准荷载

        # 简单线性模型（实际中可用更复杂的公式）
        return length_factor * (0.1 + 0.08 * load_factor)

    def step(self, action: int):
        self._decode_action(action)
        self._current_step_count += 1

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
            info["error"] = "Failed to generate valid structural model or PyG data from action."
            return next_state, reward, terminated, truncated, info

        current_model_volume = self._calculate_model_volume()
        # Perform OpenSees Analysis
        try:

            if max_indicators:
                max_disp_abs = max_indicators["max_abs_displacement"]
                max_stress = max_indicators["max_abs_stress"]
                reward = self._calculate_reward(max_disp_abs, max_stress, current_model_volume)

                self._last_max_disp_abs = max_disp_abs
                self._last_max_stress = max_stress
                self._last_model_volume = current_model_volume

                info["max_displacement_mm"] = max_disp_abs
                info["max_stress"] = max_stress
                info["model_volume_m3"] = current_model_volume
                info["frame_length"] = self.current_params["length"]

                if max_disp_abs < self.max_disp and max_stress < self.max_stress:
                    terminated = True

            else:
                reward = -10
                terminated = True
                info["error"] = "OpenSees static analysis failed."

        except Exception as e:
            print(f"Error during OpenSees analysis in env.step: {e}")
            # import traceback; traceback.print_exc() # For detailed debugging
            reward = -250.0
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
