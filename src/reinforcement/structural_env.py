from typing import Dict, Optional

import gym
import numpy as np
from gym import spaces

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.graph.structural_graph import BeamAsEdgeGraph, BeamAsNodeGraph
from src.structure.mesh_generator import MeshGenerator, MeshParameters  # For loading initial full config
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

    def __init__(self, initial_params_config_path: str, seed: int = None):
        super(StructuralEnv, self).__init__()
        self.logger = Logger("rewards")
        self.seed = seed
        self._reset_count = 0

        self.section_library = SectionLibrary()  # Assumes default json path or configured
        self.opensees_analyzer = OpenSeesAnalyzer()

        # Load initial full structural configuration
        initial_full_config = load_config(initial_params_config_path)
        analysis_config = initial_full_config["analysis_params"]
        self.max_stress = analysis_config["max_stress"]
        self.max_disp = analysis_config["max_disp"]
        self.env_config = initial_full_config["env_config"]
        # This base_config_dict contains all parameters, including fixed ones (length, width, etc.)
        # and initial values for the optimizable ones.
        self.base_config_dict = initial_full_config["structural_params"]

        # Initialize current optimizable parameters from this initial config
        # These will be updated by agent actions
        self.current_length = self.base_config_dict["length"]
        self.current_main_beam_spacing = self.base_config_dict["main_beam_spacing"]
        self.current_secondary_beam_spacing = self.base_config_dict["main_beam_spacing"]
        self.current_brace_spacing = self.base_config_dict["brace_spacing"]
        self.current_frame_section_index = self.base_config_dict["frame_section_index"]
        self.current_main_beam_section_index = self.base_config_dict["main_beam_section_index"]
        self.current_secondary_beam_section_index = self.base_config_dict["secondary_beam_section_index"]
        self.current_brace_section_index = self.base_config_dict["brace_section_index"]
        self.current_pressure = self.base_config_dict["pressure"]

        self.current_structural_model: StructuralModel | None = None
        self.current_pyg_data_state = None  # Will be a PyG Data object

        # For tracking episode length or other conditions
        self._max_steps_per_episode = initial_full_config["rl_params"]["max_steps_per_episode"]
        self._current_step_count = 0

        # Perform initial reset to set up the first state
        self._last_max_disp_abs = 0.0
        self._last_max_stress = 0.0
        self._last_model_volume = 0.0
        self.reset()  # Called by typical RL training loop setup

    def _get_current_structural_parameters_obj(self) -> StructuralParameters:
        """
        Creates a StructuralParameters object using the base fixed parameters
        and the current values of the optimizable parameters.
        """
        # Start with a copy of the initial full parameter dictionary (containing L, W, H, windows etc.)
        params_data_dict = self.base_config_dict.copy()

        # Override with current optimizable values
        params_data_dict["length"] = self.current_length
        params_data_dict["main_beam_spacing"] = self.current_main_beam_spacing
        params_data_dict["secondary_beam_spacing"] = self.current_main_beam_spacing
        params_data_dict["brace_spacing"] = self.current_brace_spacing
        params_data_dict["frame_section_index"] = self.current_frame_section_index
        params_data_dict["main_beam_section_index"] = self.current_main_beam_section_index
        params_data_dict["secondary_beam_section_index"] = self.current_secondary_beam_section_index
        params_data_dict["brace_section_index"] = self.current_brace_section_index

        # Re-create Window objects from the config dictionary format
        windows_list = []
        if "windows" in params_data_dict and params_data_dict["windows"]:
            for w_config in params_data_dict["windows"]:
                # Ensure all keys expected by Window dataclass are present or have defaults
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

        # Create StructuralParameters object using all keys from the modified dict
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
                graph_converter = BeamAsNodeGraph(self.current_structural_model)
            elif self.env_config["graph_type"] == "beam_as_edge":
                graph_converter = BeamAsEdgeGraph(self.current_structural_model)
            else:
                raise ValueError(f"Invalid graph type: {self.env_config['graph_type']}")

            env_info = {
                "frame_length": self.current_length,
            }
            self.current_pyg_data_state = graph_converter.to_pyg_data(env_info)
        except Exception as e:
            print(f"Error during graph conversion in _generate_model_and_state_from_current_params: {e}")
            return False, None

        return True, max_indicators

    def reset(self, options=None):
        super().reset(seed=self.seed + self._reset_count)
        self._reset_count += 1

        # init section index randomly
        random_params = RandomStructure(self.section_library, self._np_random).generate_random_parameters()

        self.current_length = random_params.length
        self.current_frame_section_index = random_params.frame_section_index
        self.current_main_beam_section_index = random_params.main_beam_section_index
        self.current_secondary_beam_section_index = random_params.secondary_beam_section_index
        self.current_brace_section_index = random_params.brace_section_index
        # self.current_pressure = random_params.pressure

        self.current_main_beam_spacing = random_params.main_beam_spacing
        self.current_secondary_beam_spacing = random_params.secondary_beam_spacing
        self.current_brace_spacing = random_params.brace_spacing

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

        info = {
            "max_displacement_mm": self._last_max_disp_abs,
            "max_stress": self._last_max_stress,
            "model_volume_m3": self._last_model_volume,
            "frame_length": self.current_length,
        }
        return self.current_pyg_data_state, info

    def _decode_action(self, action_vector: np.ndarray):
        """
        Decodes the continuous action vector from the agent (assumed to be in [0, 1])
        into actual structural parameter values.
        """
        param_map = self.env_config["parameter_action_mapping"]

        # Main Beam Spacing
        cfg = param_map["main_beam_spacing"]
        raw_val = cfg["bounds"][0] + (1 - action_vector[cfg["action_idx"]]) * (
            cfg["bounds"][1] - cfg["bounds"][0]
        )
        self.current_main_beam_spacing = np.clip(
            round(raw_val / cfg["module"]) * cfg["module"], cfg["bounds"][0], cfg["bounds"][1]
        )

        # Brace Spacing
        cfg = param_map["brace_spacing"]
        raw_val = cfg["bounds"][0] + action_vector[cfg["action_idx"]] * (cfg["bounds"][1] - cfg["bounds"][0])
        num_spans = np.clip(round(raw_val), cfg["bounds"][0], cfg["bounds"][1])
        if num_spans > 1:
            brace_spacing = (self.current_main_beam_spacing / num_spans).round(3)
        else:
            brace_spacing = (1.3 * (self.current_main_beam_spacing / 2)).round(3)
        self.current_brace_spacing = brace_spacing if brace_spacing > 0.3 else 0.3

        def _decode_single_action(section_index_name):
            cfg = param_map[section_index_name]
            num_options = cfg["num_options"]
            ## 1. interpret action as normalized index of the section
            raw_idx = action_vector[cfg["action_idx"]] * (num_options - 1)  # Scale [0,1] to [0, N-1]
            return np.clip(int(round(raw_idx)), 0, num_options - 1)

            # ### 2. interpret action as increase or decrease of the section index
            # current_section_index = getattr(self, f"current_{section_index_name}")
            # if action_vector[cfg["action_idx"]] > 0.5:
            #     updated_section_index = current_section_index + 1
            # else:
            #     updated_section_index = current_section_index - 1
            # return np.clip(updated_section_index, 0, num_options - 1)

        # 4. Frame Section Index
        self.current_frame_section_index = _decode_single_action("frame_section_index")

        # 5. Main Beam Section Index
        self.current_main_beam_section_index = _decode_single_action("main_beam_section_index")

        # 6. Secondary Beam Section Index
        self.current_secondary_beam_section_index = _decode_single_action("secondary_beam_section_index")

        # 7. Brace Section Index
        self.current_brace_section_index = _decode_single_action("brace_section_index")

    def _run_fem_analysis(self) -> Optional[Dict[str, float]]:
        # create mesh
        mesh_params = MeshParameters(max_grid_size=0.2)
        mesh_generator = MeshGenerator(self.current_structural_model, mesh_params)
        mesh_generator.generate_mesh()
        meshed_model = mesh_generator.meshed_model

        self.opensees_analyzer.setup_opensees_model(mesh_generator.meshed_model)
        self.opensees_analyzer.apply_gravity_and_pressure_loads(self.current_pressure * 1e3)
        static_success = self.opensees_analyzer.run_static_analysis()
        if not static_success:
            return None

        # 填充梁端节点荷载
        nodal_loads = self.current_structural_model.calc_node_load(self.current_pressure * 1e3)
        for beam_id, beam_obj in self.current_structural_model.beams.items():
            beam_obj.start_node_load = nodal_loads[beam_obj.start_node]
            beam_obj.end_node_load = nodal_loads[beam_obj.end_node]

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

    def _calculate_reward(self, max_disp_abs: float, max_stress: float, model_volume: float):
        disp_constraint = self.max_disp
        stress_constraint = self.max_stress

        if max_disp_abs < disp_constraint:
            delta_c1 = 1.0
        else:
            delta_c1 = (
                (self._last_max_disp_abs - max_disp_abs)
                / disp_constraint
                / (max_disp_abs / disp_constraint + 1)
            )

        if max_stress < stress_constraint:
            delta_c2 = 1.0
        else:
            delta_c2 = (
                (self._last_max_stress - max_stress)
                / stress_constraint
                / (max_stress / stress_constraint + 1)
            )

        delta_v = model_volume - self._last_model_volume

        if delta_c1 == 1 and delta_c2 == 1:
            reward = -delta_v * 100 if delta_v < 0 else -delta_v * 200
        else:
            reward = -(2 - delta_c1 - delta_c2) * 10

        return reward

    def step(self, action: np.ndarray):
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
                info["frame_length"] = self.current_length

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
        print(f"Step: {self._current_step_count}")
        print("Frame Length: ", self.current_length, "Pressure: ", self.current_pressure)
        print(
            f"  Main Spacing: {self.current_main_beam_spacing:.2f}, Secondary Spacing: {self.current_main_beam_spacing:.2f}, Brace Spacing: {self.current_brace_spacing:.2f}"
        )
        print(
            f"  Frame Idx: {self.current_frame_section_index}, Main Idx: {self.current_main_beam_section_index}, Secondary Idx: {self.current_secondary_beam_section_index}, Brace Idx: {self.current_brace_section_index}"
        )
        print(
            f"  Max Disp: {self._last_max_disp_abs:.2f}, Max Stress: {self._last_max_stress:.2f}, Model Volume: {self._last_model_volume:.2f}"
        )
        # print section info
        print(
            "Frame Section:",
            self.section_library.sections["方钢"][self.current_frame_section_index].name,
        )
        print(
            "Main Beam Section:",
            self.section_library.sections["H型钢-主梁"][self.current_main_beam_section_index].name,
        )
        print(
            "Secondary Beam Section:",
            self.section_library.sections["H型钢-次梁"][self.current_secondary_beam_section_index].name,
        )
        print(
            "Brace Section:",
            self.section_library.sections["C型钢"][self.current_brace_section_index].name,
        )

    def close(self):
        # Clean up any resources if needed (e.g., OpenSees model if ops.wipe() isn't sufficient)
        print("StructuralEnv closed.")


if __name__ == "__main__":
    env = StructuralEnv("config.yaml", seed=123)
    for i in range(3):
        env.reset()
        env.render()
        print("-" * 100)
