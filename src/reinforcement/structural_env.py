from typing import Dict, Optional

import gym
import numpy as np
import torch  # Only if PyG Data objects are directly returned and type hinted
from gym import spaces

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.graph.structural_graph import BeamAsNodeGraph
from src.structure.structural_model import SectionLibrary, StructuralModel, StructuralParameters, Window
from src.utils.model_utils import load_config  # For loading initial full config


class StructuralEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, initial_params_config_path: str, env_action_config: dict):
        super(StructuralEnv, self).__init__()

        self.env_action_config = env_action_config
        self.section_library = SectionLibrary()  # Assumes default json path or configured
        self.opensees_analyzer = OpenSeesAnalyzer()

        # Load initial full structural configuration
        initial_full_config = load_config(initial_params_config_path)
        # This base_config_dict contains all parameters, including fixed ones (length, width, etc.)
        # and initial values for the optimizable ones.
        self.base_config_dict = initial_full_config["structural_params"]

        # Initialize current optimizable parameters from this initial config
        # These will be updated by agent actions
        self.current_main_beam_spacing = self.base_config_dict["main_beam_spacing"]
        self.current_secondary_beam_spacing = self.base_config_dict["secondary_beam_spacing"]
        self.current_brace_spacing = self.base_config_dict["brace_spacing"]
        self.current_frame_section_index = self.base_config_dict["frame_section_index"]
        self.current_main_beam_section_index = self.base_config_dict["main_beam_section_index"]
        self.current_secondary_beam_section_index = self.base_config_dict["secondary_beam_section_index"]
        self.current_brace_section_index = self.base_config_dict["brace_section_index"]

        # Define action space (7 continuous values, agent output assumed in [0, 1])
        self.action_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        # Define observation space (placeholder as agent uses PyG Data directly)
        # For strict Gym compliance, this would need to be defined based on flattened graph features
        # or a complex Dict space.
        self.observation_space = spaces.Dict({})

        self.current_structural_model: StructuralModel | None = None
        self.current_pyg_data_state = None  # Will be a PyG Data object

        # For tracking episode length or other conditions
        self._max_steps_per_episode = env_action_config.get("max_steps", 200)
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
        params_data_dict["main_beam_spacing"] = self.current_main_beam_spacing
        params_data_dict["secondary_beam_spacing"] = self.current_secondary_beam_spacing
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
                    "width": w_config.get("width", 1.0),
                    "height": w_config.get("height", 1.0),
                    "bottom_height": w_config.get("bottom_height", 0.0),
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

        if not self.current_structural_model.beams:
            print("Warning (StructuralEnv): Generated structural model has no beams.")
            # Create an empty graph state representation
            # Note: rl_agent might expect certain attributes even on an "empty" graph
            self.current_pyg_data_state = BeamAsNodeGraph(self.current_structural_model).to_pyg_data()
            return False

        graph_converter = BeamAsNodeGraph(self.current_structural_model)
        self.current_pyg_data_state = graph_converter.to_pyg_data()

        if self.current_pyg_data_state.num_nodes == 0 and len(self.current_structural_model.beams) > 0:
            print("Warning (StructuralEnv): PyG data has 0 nodes even though model has beams.")
            # This case might indicate an issue in graph conversion for certain structures
            return False
        elif self.current_pyg_data_state.num_nodes == 0:
            print("Info (StructuralEnv): PyG data has 0 nodes (likely no beams in model).")

        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # For gym compatibility if using random seeding later

        # Reset optimizable parameters to their initial values from the base config
        self.current_main_beam_spacing = self.base_config_dict["main_beam_spacing"]
        self.current_secondary_beam_spacing = self.base_config_dict["secondary_beam_spacing"]
        self.current_brace_spacing = self.base_config_dict["brace_spacing"]
        self.current_frame_section_index = self.base_config_dict["frame_section_index"]
        self.current_main_beam_section_index = self.base_config_dict["main_beam_section_index"]
        self.current_secondary_beam_section_index = self.base_config_dict["secondary_beam_section_index"]
        self.current_brace_section_index = self.base_config_dict["brace_section_index"]

        self._current_step_count = 0

        generation_success = self._generate_model_and_state_from_current_params()
        if not generation_success:
            print("Critical Error (StructuralEnv): Failed to generate a valid model/state at reset.")
            raise ValueError("Failed to generate a valid model/state at reset.")

        max_indicators = self._run_fem_analysis()
        if max_indicators:
            self._last_max_disp_abs = max_indicators["max_displacement"]
            self._last_max_stress = max_indicators["max_stress"]
        else:
            raise ValueError("Failed to run FEM analysis at reset.")

        self._last_model_volume = self._calculate_model_volume()

        info = {
            "max_displacement_mm": self._last_max_disp_abs,
            "max_stress": self._last_max_stress,
            "model_volume_m3": self._last_model_volume,
        }  # Empty info dict for reset
        return self.current_pyg_data_state, info

    def _decode_action(self, action_vector: np.ndarray):
        """
        Decodes the continuous action vector from the agent (assumed to be in [0, 1])
        into actual structural parameter values.
        """
        param_map = self.env_action_config["parameter_action_mapping"]

        # # 1. Main Beam Spacing
        # cfg = param_map["main_beam_spacing"]
        # raw_val = cfg["bounds"][0] + action_vector[cfg["action_idx"]] * (cfg["bounds"][1] - cfg["bounds"][0])
        # self.current_main_beam_spacing = np.clip(
        #     round(raw_val / cfg["module"]) * cfg["module"], cfg["bounds"][0], cfg["bounds"][1]
        # )

        # # 2. Secondary Beam Spacing
        # cfg = param_map["secondary_beam_spacing"]
        # raw_val = cfg["bounds"][0] + action_vector[cfg["action_idx"]] * (cfg["bounds"][1] - cfg["bounds"][0])
        # self.current_secondary_beam_spacing = np.clip(
        #     round(raw_val / cfg["module"]) * cfg["module"], cfg["bounds"][0], cfg["bounds"][1]
        # )

        # # 3. Brace Spacing
        # cfg = param_map["brace_spacing"]
        # raw_val = cfg["bounds"][0] + action_vector[cfg["action_idx"]] * (cfg["bounds"][1] - cfg["bounds"][0])
        # self.current_brace_spacing = np.clip(
        #     round(raw_val / cfg["module"]) * cfg["module"], cfg["bounds"][0], cfg["bounds"][1]
        # )

        def _decode_single_action(section_index_name):
            cfg = param_map[section_index_name]
            num_options = cfg["num_options"]
            raw_idx = action_vector[cfg["action_idx"]] * (num_options - 1)  # Scale [0,1] to [0, N-1]
            return np.clip(int(round(raw_idx)), 0, num_options - 1)

        # 4. Frame Section Index
        self.current_frame_section_index = _decode_single_action("frame_section_index")

        # 5. Main Beam Section Index
        self.current_main_beam_section_index = _decode_single_action("main_beam_section_index")

        # 6. Secondary Beam Section Index
        self.current_secondary_beam_section_index = _decode_single_action("secondary_beam_section_index")

        # 7. Brace Section Index
        self.current_brace_section_index = _decode_single_action("brace_section_index")

    def _run_fem_analysis(self) -> Optional[Dict[str, float]]:
        self.opensees_analyzer.setup_opensees_model(self.current_structural_model)
        self.opensees_analyzer.apply_gravity_and_pressure_loads(30000.0)
        static_success = self.opensees_analyzer.run_static_analysis()
        if not static_success:
            return None
        max_indicators = self.opensees_analyzer.get_max_indicators()
        return max_indicators

    def _calculate_model_volume(self) -> float:
        return self.current_structural_model.calculate_mass() / 7850.0

    def _calculate_reward(self, max_disp_abs: float, max_stress: float, model_volume: float):
        disp_constraint = 25.0
        stress_constraint = 150.0

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

        reward = -delta_v * (2 - delta_c1 - delta_c2)
        return reward

    def step(self, action: np.ndarray):
        self._decode_action(action)
        self._current_step_count += 1

        generation_success = self._generate_model_and_state_from_current_params()

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
            reward = -100.0
            terminated = True
            info["error"] = "Failed to generate valid structural model or PyG data from action."
            return next_state, reward, terminated, truncated, info

        current_model_volume = self._calculate_model_volume()
        # Perform OpenSees Analysis
        try:
            max_indicators = self._run_fem_analysis()
            if max_indicators:
                max_disp_abs = max_indicators["max_displacement"]
                max_stress = max_indicators["max_stress"]
                reward = self._calculate_reward(max_disp_abs, max_stress, current_model_volume)

                info["max_displacement_mm"] = max_disp_abs
                info["max_stress"] = max_stress
                info["model_volume_m3"] = current_model_volume

            else:
                reward = -200.0
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
        print(
            f"  Main Spacing: {self.current_main_beam_spacing:.2f}, Secondary Spacing: {self.current_secondary_beam_spacing:.2f}, Brace Spacing: {self.current_brace_spacing:.2f}"
        )
        print(
            f"  Frame Idx: {self.current_frame_section_index}, Main Idx: {self.current_main_beam_section_index}, Secondary Idx: {self.current_secondary_beam_section_index}, Brace Idx: {self.current_brace_section_index}"
        )
        if self.current_structural_model:
            print(f"  Num Beams: {len(self.current_structural_model.beams)}")

    def close(self):
        # Clean up any resources if needed (e.g., OpenSees model if ops.wipe() isn't sufficient)
        print("StructuralEnv closed.")
