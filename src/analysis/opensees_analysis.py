import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import openseespy.opensees as ops

from src.structure.structural_model import Beam, StructuralModel  # For type hinting

# Conversion Factors and Constants
M_TO_MM = 1000.0
M2_TO_MM2 = 1.0e6
M4_TO_MM4 = 1.0e12
KG_M3_TO_TONNE_MM3 = 1.0e-12  # 1 kg/m^3 = 1e-12 tonne/mm^3
N_M_TO_N_MM = 1.0e-3

# Material Properties (Steel)
E_STEEL_MPA = 2.0e5  # Elastic Modulus in N/mm^2 (MPa) -> typical 200 GPa
NU_STEEL = 0.3  # Poisson's ratio
G_STEEL_MPA = E_STEEL_MPA / (2 * (1 + NU_STEEL))  # Shear Modulus in N/mm^2 (MPa)
RHO_STEEL_KG_M3 = 7850.0  # Density in kg/m^3
RHO_STEEL_TONNE_MM3 = RHO_STEEL_KG_M3 * KG_M3_TO_TONNE_MM3  # Density in tonne/mm^3

# Gravity
G_MM_S2 = 9806.65  # Gravitational acceleration in mm/s^2
G_M_S2 = 9.80665


class OpenSeesAnalyzer:
    """OpenSees分析接口类, using N, mm, s, tonne units."""

    def __init__(self):
        self.model_built = False
        self.loads_applied = False
        self.support_node_ids: List[int] = []  # Stores original model node IDs
        # model_beam_id -> (x_loc, y_loc, z_loc)
        self.beam_local_coord_systems: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _calculate_beam_local_axes_and_vecxz(
        self, p1_coords_mm: np.ndarray, p2_coords_mm: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculates local x, y, z axes for a beam and the vecxz for geomTransf.
        Returns (x_local, y_local, z_local, vecxz)
        vecxz is a vector in the global system that defines the local xz-plane of the element.
        It must be perpendicular to the element's local x-axis.
        """
        dx_vec = p2_coords_mm - p1_coords_mm
        length = np.linalg.norm(dx_vec)
        if length < 1e-6:  # Zero length beam
            return None, None, None, None

        x_local = dx_vec / length

        # Default vecxz for vertical members (local x-z plane aligned with global X-Z)
        if np.isclose(abs(x_local[2]), 1.0):  # Beam is vertical (aligned with global Z)
            # Local z-axis can be chosen along global X to define the xz-plane
            vecxz = np.array([1.0, 0.0, 0.0])
            y_local = np.cross(vecxz, x_local)  # Should be -global Y or +global Y
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)  # This will be the actual local z, aligned with vecxz
        else:  # Beam is not vertical
            global_z_axis = np.array([0.0, 0.0, 1.0])
            # Try to make local z-axis point upwards in the plane perpendicular to beam's x-axis
            # Project global Z onto the plane normal to x_local
            proj_global_z_on_x_local = np.dot(global_z_axis, x_local) * x_local
            z_local_component_in_plane = global_z_axis - proj_global_z_on_x_local

            if np.linalg.norm(z_local_component_in_plane) < 1e-6:
                # Beam is horizontal (x_local is in XY plane). Local z is along global Z.
                # Or beam is aligned with global Z (already handled by the first `if`)
                # This case means x_local is in XY plane, so local z is global Z.
                z_local = global_z_axis
                # vecxz is typically the local z-axis vector
                vecxz = z_local
                y_local = np.cross(z_local, x_local)
            else:
                z_local = z_local_component_in_plane / np.linalg.norm(z_local_component_in_plane)
                # vecxz is typically the local z-axis vector
                vecxz = z_local
                y_local = np.cross(z_local, x_local)

        return x_local, y_local, z_local, vecxz

    def setup_opensees_model(self, model: StructuralModel):
        """
        Creates the OpenSees model including nodes, materials, elements,
        boundary conditions, and nodal masses for modal analysis.
        Units: N, mm, s, tonne.
        """
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)  # 3 dimensions, 6 DOFs per node

        # Create Nodes
        node_coords_mm_map: Dict[int, np.ndarray] = {}  # node_id -> coords_mm
        for node_id, point_obj in model.points.items():
            coords_mm = point_obj.coordinates() * M_TO_MM
            ops.node(node_id, *coords_mm)  # Use model node_id directly as OpenSees tag
            node_coords_mm_map[node_id] = coords_mm

        # Geometric Transformations and Elements
        geom_transf_counter = 0  # Start geomTransf tags from 0 (or 1 if preferred)

        for beam_id, beam_obj in model.beams.items():
            start_node_id = beam_obj.start_node
            end_node_id = beam_obj.end_node

            p1_coords = node_coords_mm_map[start_node_id]
            p2_coords = node_coords_mm_map[end_node_id]

            x_loc, y_loc, z_loc, vecxz = self._calculate_beam_local_axes_and_vecxz(p1_coords, p2_coords)
            if vecxz is None:
                print(f"Warning: Skipping zero-length beam (model ID {beam_id}).")
                continue

            self.beam_local_coord_systems[beam_id] = (x_loc, y_loc, z_loc)

            current_geom_transf_tag = (
                geom_transf_counter  # Assign a unique tag for each element's transformation
            )
            ops.geomTransf("Linear", current_geom_transf_tag, *vecxz)

            section_props = model.sections.get(beam_obj.section)
            if not section_props:
                print(f"Warning: Section {beam_obj.section} for beam {beam_id} not found. Skipping beam.")
                continue

            A_mm2 = section_props.area * M2_TO_MM2
            Iz_local_mm4 = section_props.Ix * M4_TO_MM4  # Bending about local z-axis
            Iy_local_mm4 = section_props.Iy * M4_TO_MM4  # Bending about local y-axis
            J_torsion_mm4 = section_props.Iz * M4_TO_MM4  # Torsional Moment of Inertia

            # Use model beam_id, start_node_id, end_node_id directly as OpenSees tags
            ops.element(
                "elasticBeamColumn",
                beam_id,
                start_node_id,
                end_node_id,
                A_mm2,
                E_STEEL_MPA,
                G_STEEL_MPA,
                J_torsion_mm4,
                Iy_local_mm4,
                Iz_local_mm4,
                current_geom_transf_tag,
            )
            geom_transf_counter += 1

        # Boundary Conditions: Fix bottom 4 outermost nodes
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        bottom_nodes_data: List[Tuple[int, np.ndarray]] = []  # (node_id, coords_mm)

        for node_id, point_obj in model.points.items():  # Using model.points
            if np.isclose(point_obj.z, 0.0):  # Bottom nodes
                coords_mm = node_coords_mm_map[node_id]
                bottom_nodes_data.append((node_id, coords_mm))
                min_x = min(min_x, coords_mm[0])
                max_x = max(max_x, coords_mm[0])
                min_y = min(min_y, coords_mm[1])
                max_y = max(max_y, coords_mm[1])

        self.support_node_ids = []
        corner_nodes_found = 0
        if bottom_nodes_data:  # Ensure there are bottom nodes before proceeding
            potential_corners = []
            for node_id, coords in bottom_nodes_data:
                is_min_x = np.isclose(coords[0], min_x)
                is_max_x = np.isclose(coords[0], max_x)
                is_min_y = np.isclose(coords[1], min_y)
                is_max_y = np.isclose(coords[1], max_y)
                if (
                    (is_min_x and is_min_y)
                    or (is_max_x and is_min_y)
                    or (is_min_x and is_max_y)
                    or (is_max_x and is_max_y)
                ):
                    potential_corners.append(node_id)

            # Select up to 4 unique corners, prioritizing the exact corner matches
            # This simple selection takes the first few found; more robust selection might be needed for complex bases
            for node_id in potential_corners:
                if corner_nodes_found < 4 and node_id not in self.support_node_ids:
                    ops.fix(node_id, 1, 1, 1, 1, 1, 1)  # Fix all 6 DOFs, using model node_id
                    self.support_node_ids.append(node_id)
                    corner_nodes_found += 1

        if corner_nodes_found != 4 and len(model.points) > 0:  # Check if any nodes exist
            print(
                f"Warning: Expected to find 4 corner support nodes, but fixed {corner_nodes_found}. Bottom nodes found: {len(bottom_nodes_data)}. "
            )

        # Nodal Masses
        nodal_masses_tonne: Dict[int, np.ndarray] = {nid: np.zeros(6) for nid in model.points.keys()}

        for beam_id, beam_obj in model.beams.items():
            section_props = model.sections.get(beam_obj.section)
            if not section_props:
                continue

            start_node_model = model.points[beam_obj.start_node]
            end_node_model = model.points[beam_obj.end_node]
            length_m = start_node_model.distance_to(end_node_model)
            length_mm = length_m * M_TO_MM

            area_mm2 = section_props.area * M2_TO_MM2
            beam_mass_tonne = area_mm2 * length_mm * RHO_STEEL_TONNE_MM3
            mass_per_node = beam_mass_tonne / 2.0

            start_node_id = beam_obj.start_node
            end_node_id = beam_obj.end_node

            for i in range(3):  # Add to translational mass components
                nodal_masses_tonne[start_node_id][i] += mass_per_node
                nodal_masses_tonne[end_node_id][i] += mass_per_node

        for node_id, mass_values in nodal_masses_tonne.items():
            if np.sum(mass_values[:3]) > 1e-9:  # Only add mass if it's significant
                ops.mass(node_id, *mass_values)  # Use model node_id directly

        self.model_built = True
        print("OpenSees model setup complete.")

    def apply_gravity_loads(self, model: StructuralModel):
        """Applies gravity load as uniform beam loads in OpenSees."""
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup_opensees_model first.")

        ops.timeSeries("Linear", 1)  # Load Pattern TimeSeries Tag
        ops.pattern("Plain", 1, 1)  # Load Pattern Tag, TimeSeries Tag

        for beam_id, beam_obj in model.beams.items():
            section_props = model.sections.get(beam_obj.section)
            if not section_props:
                continue

            area_m2 = section_props.area
            w_N_per_m = area_m2 * RHO_STEEL_KG_M3 * G_M_S2
            w_N_per_mm = w_N_per_m * N_M_TO_N_MM

            if beam_id not in self.beam_local_coord_systems:
                print(f"Warning: Local axes not found for beam {beam_id}. Skipping gravity load.")
                continue

            x_loc, y_loc, z_loc = self.beam_local_coord_systems[beam_id]
            R_glob_to_loc = np.array([x_loc, y_loc, z_loc])
            F_global_unit_force = np.array([0.0, 0.0, -w_N_per_mm])
            F_local_components = R_glob_to_loc @ F_global_unit_force

            load_val_local_x = F_local_components[0]
            load_val_local_y = F_local_components[1]
            load_val_local_z = F_local_components[2]

            # Use model beam_id directly as OpenSees ele tag
            ops.eleLoad(
                "-ele", beam_id, "-type", "-beamUniform", load_val_local_y, load_val_local_z, load_val_local_x
            )

        self.loads_applied = True
        print("Gravity loads applied.")

    def run_static_analysis(self) -> bool:
        """Performs static analysis under the applied loads."""
        if not self.model_built:
            raise RuntimeError("Model not built.")
        if not self.loads_applied:
            print("Warning: No loads seem to have been applied before static analysis.")

        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")

        print("Running static analysis...")
        result = ops.analyze(1)
        if result == 0:
            print("Static analysis successful.")
        else:
            print(f"Static analysis FAILED. Result code: {result}")
        return result == 0

    def run_modal_analysis(self, num_modes: int) -> Optional[List[float]]:
        """Performs modal analysis."""
        if not self.model_built:
            raise RuntimeError("Model not built.")

        print(f"Running modal analysis for {num_modes} modes...")
        eigenvalues_omega_sq = ops.eigen(num_modes)
        if (
            eigenvalues_omega_sq is None
            or not isinstance(eigenvalues_omega_sq, list)
            or len(eigenvalues_omega_sq) == 0
        ):
            print("Modal analysis failed or returned no eigenvalues.")
            return None

        print("Modal analysis successful.")
        return eigenvalues_omega_sq

    def get_node_displacements(self, query_node_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        Retrieves displacements for specified model node IDs (or all if None).
        Returns Dict[model_node_id, np.array_of_6_displacements_mm_rad]
        """
        displacements: Dict[int, np.ndarray] = {}
        target_node_ids_to_query = []

        if query_node_ids is None:  # Get for all nodes defined in the OpenSees model
            # ops.getNodeTags() returns all tags known to OpenSees
            # Assuming these are the same 0-indexed IDs we used for creation.
            target_node_ids_to_query = ops.getNodeTags()
        else:
            target_node_ids_to_query = query_node_ids

        for node_id in target_node_ids_to_query:
            try:
                # Use model node_id directly as OpenSees tag
                disp_mm_rad = np.array(ops.nodeDisp(node_id))
                displacements[node_id] = disp_mm_rad
            except Exception as e:
                print(f"Could not get displacement for OpenSees node tag {node_id}: {e}")
        return displacements

    def get_support_reactions(self) -> Dict[int, np.ndarray]:
        """
        Retrieves reactions for the defined support nodes.
        Returns Dict[model_node_id, np.array_of_6_reactions_N_Nmm]
        """
        reactions: Dict[int, np.ndarray] = {}
        if not self.support_node_ids:
            print("Warning: No support nodes defined/found to get reactions from.")
            return reactions

        ops.reactions()  # Ensure reactions are calculated/updated

        for node_id in self.support_node_ids:  # Iterate original model IDs stored
            try:
                # Use model node_id directly as OpenSees tag
                react_N_Nmm = np.array(ops.nodeReaction(node_id))
                reactions[node_id] = react_N_Nmm
            except Exception as e:
                print(f"Could not get reaction for OpenSees support node tag {node_id}: {e}")
        return reactions

    def save_results(self, data: Dict, filename: str, output_dir: str = "results"):
        """Saves dictionary data to a JSON file in the specified directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_data[key] = [i.tolist() for i in value]
            elif isinstance(value, dict):
                inner_serializable_data = {}
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, np.ndarray):
                        inner_serializable_data[str(inner_key)] = inner_value.tolist()
                    else:
                        inner_serializable_data[str(inner_key)] = inner_value
                serializable_data[key] = inner_serializable_data
            else:
                serializable_data[key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Results saved to {filepath}")

    def perform_full_analysis_and_save(
        self, model: StructuralModel, num_modal_modes: int = 5, output_dir: str = "opensees_results"
    ):
        """
        Convenience method to run the full sequence: setup, load, static & modal analysis, save results.
        """
        print("Starting OpenSees full analysis workflow...")
        self.setup_opensees_model(model)

        if self.model_built:
            self.apply_gravity_loads(model)

            static_success = self.run_static_analysis()
            if static_success:
                displacements = self.get_node_displacements()
                reactions = self.get_support_reactions()

                static_results_data = {
                    "displacements_mm_rad": displacements,
                    "support_reactions_N_Nmm": reactions,
                }
                self.save_results(static_results_data, "static_analysis_results.json", output_dir)

            eigenvalues_rad2_s2 = self.run_modal_analysis(num_modal_modes)
            if eigenvalues_rad2_s2:
                frequencies_hz = [np.sqrt(val) / (2 * np.pi) for val in eigenvalues_rad2_s2 if val > 0]
                periods_s = [1 / f for f in frequencies_hz if f > 0]
                modal_results_data = {
                    "eigenvalues_omega_sq_rad2_s2": eigenvalues_rad2_s2,
                    "frequencies_hz": frequencies_hz,
                    "periods_s": periods_s,
                }
                self.save_results(modal_results_data, "modal_analysis_results.json", output_dir)

        print("OpenSees full analysis workflow finished.")


# Example Usage (can be in a main script or a test file)
if __name__ == "__main__":
    from src.utils.model_utils import create_structural_model, load_config

    print("Testing OpenSeesAnalyzer...")
    config_path = "config.yaml"
    config = load_config(config_path)
    model = create_structural_model(config)

    analyzer = OpenSeesAnalyzer()
    try:
        analyzer.perform_full_analysis_and_save(
            model, num_modal_modes=6, output_dir="results/opensees_test_results_0idx"
        )

    except Exception as e:
        print(f"An error occurred during OpenSees analysis: {e}")
        import traceback

        traceback.print_exc()
