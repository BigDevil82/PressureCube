import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv

from src.structure.section_library import RectangularTubeSection
from src.structure.structural_model import Beam, Plate, StructuralModel  # For type hinting
from src.utils.logger import Logger

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

logger = Logger("opensees_analysis", disable=True)


class OpenSeesAnalyzer:
    """OpenSees分析接口类, using N, mm, s, tonne units."""

    def __init__(self):
        self._model: Optional[StructuralModel] = None
        self.model_built = False
        self._static_analysis_success = False
        self.loads_applied = False
        self.support_node_ids: List[int] = []  # Stores original model node IDs
        # model_beam_id -> (x_loc, y_loc, z_loc)
        self.beam_local_coord_systems: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Plate-related attributes
        self.plate_thickness_mm = None
        self.plate_material_tag = 1000  # Starting tag for plate materials
        self.plate_section_tag = 2000  # Starting tag for plate sections
        self.pressure_load_applied = False

    def _calculate_beam_local_axes_and_vecxz(
        self, p1_coords_mm: np.ndarray, p2_coords_mm: np.ndarray, beam_id: int, beam_obj: Beam
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

        if np.isclose(abs(x_local[0]), 1.0) and beam_obj.on_face in ["top", "bottom"]:
            vecxz = np.array([0.0, 1.0, 0.0])
        elif np.isclose(abs(x_local[0]), 1.0) and beam_obj.on_face in ["left", "right"]:
            vecxz = np.array([0.0, 0.0, 1.0])
        elif np.isclose(abs(x_local[1]), 1.0) and beam_obj.on_face in ["top", "bottom"]:
            vecxz = np.array([1.0, 0.0, 0.0])
        elif np.isclose(abs(x_local[1]), 1.0) and beam_obj.on_face in ["front", "back"]:
            vecxz = np.array([0.0, 0.0, 1.0])
        elif np.isclose(abs(x_local[2]), 1.0) and beam_obj.on_face in ["front", "back"]:
            vecxz = np.array([0.0, 1.0, 0.0])
        elif np.isclose(abs(x_local[2]), 1.0) and beam_obj.on_face in ["left", "right"]:
            vecxz = np.array([1.0, 0.0, 0.0])
            # global_z_axis = np.array([0.0, 0.0, 1.0])
            # # Try to make local z-axis point upwards in the plane perpendicular to beam's x-axis
            # # Project global Z onto the plane normal to x_local
            # proj_global_z_on_x_local = np.dot(global_z_axis, x_local) * x_local
            # z_local_component_in_plane = global_z_axis - proj_global_z_on_x_local

            # if np.linalg.norm(z_local_component_in_plane) < 1e-6:
            #     # Beam is horizontal (x_local is in XY plane). Local z is along global Z.
            #     # Or beam is aligned with global Z (already handled by the first `if`)
            #     # This case means x_local is in XY plane, so local z is global Z.
            #     z_local = global_z_axis
            #     # vecxz is typically the local z-axis vector
            #     vecxz = z_local
            #     y_local = np.cross(z_local, x_local)
            # else:
            #     z_local = z_local_component_in_plane / np.linalg.norm(z_local_component_in_plane)
            #     # vecxz is typically the local z-axis vector
            #     vecxz = z_local
            #     y_local = np.cross(z_local, x_local)

        y_local = np.cross(vecxz, x_local)  # Should be -global Y or +global Y
        y_local = y_local / np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)  # This will be the actual local z, aligned with vecxz
        return x_local, y_local, z_local, vecxz

    def setup_opensees_model(self, model: StructuralModel):
        """
        Creates the OpenSees model including nodes, materials, elements,
        boundary conditions, and nodal masses for modal analysis.
        Now includes steel plate modeling with ShellMITC4 elements.
        Units: N, mm, s, tonne.

        Args:
            model: StructuralModel object containing beams and plates
            plate_thickness_mm: Thickness of steel plates in mm (default 10mm)
        """
        # model.plates = None
        self.plate_thickness_mm = model.params.plate_thickness * M_TO_MM
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)  # 3 dimensions, 6 DOFs per node

        # 确保节点数与梁包含的节点数相同
        beam_nodes = set()
        for beam_id, beam_obj in model.beams.items():
            beam_nodes.add(beam_obj.start_node)
            beam_nodes.add(beam_obj.end_node)
        assert len(beam_nodes) == len(model.points), "节点数与梁构件数量不一致"

        # Create Nodes
        node_coords_mm_map: Dict[int, np.ndarray] = {}  # node_id -> coords_mm
        for node_id, point_obj in model.points.items():
            coords_mm = point_obj.coordinates() * M_TO_MM
            ops.node(node_id, *coords_mm)  # Use model node_id directly as OpenSees tag
            node_coords_mm_map[node_id] = coords_mm

        # Create plate materials and sections
        if model.plates:
            self._create_plate_materials_and_sections(self.plate_thickness_mm)

        # Geometric Transformations and Beam Elements
        geom_transf_counter = 0  # Start geomTransf tags from 0 (or 1 if preferred)

        for beam_id, beam_obj in model.beams.items():
            start_node_id = beam_obj.start_node
            end_node_id = beam_obj.end_node

            p1_coords = node_coords_mm_map[start_node_id]
            p2_coords = node_coords_mm_map[end_node_id]

            x_loc, y_loc, z_loc, vecxz = self._calculate_beam_local_axes_and_vecxz(
                p1_coords, p2_coords, beam_id, beam_obj
            )
            if vecxz is None:
                logger.warning(f"Warning: Skipping zero-length beam (model ID {beam_id}).")
                continue

            self.beam_local_coord_systems[beam_id] = (x_loc, y_loc, z_loc)

            # Assign a unique tag for each element's transformation
            current_geom_transf_tag = geom_transf_counter
            ops.geomTransf("Linear", current_geom_transf_tag, *vecxz)

            section_props = model.sections.get(beam_obj.section)
            if not section_props:
                logger.warning(
                    f"Warning: Section {beam_obj.section} for beam {beam_id} not found. Skipping beam."
                )
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

        # Add plate elements
        if model.plates:
            self._add_plate_elements(model)

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
            print(f"potential_corners: {potential_corners}")
            for node_id in potential_corners:
                if corner_nodes_found < 4 and node_id not in self.support_node_ids:
                    ops.fix(node_id, 1, 1, 1, 1, 1, 1)  # Fix all 6 DOFs, using model node_id
                    self.support_node_ids.append(node_id)
                    corner_nodes_found += 1

        if corner_nodes_found != 4 and len(model.points) > 0:  # Check if any nodes exist
            logger.warning(
                f"Warning: Expected to find 4 corner support nodes, but fixed {corner_nodes_found}. Bottom nodes found: {len(bottom_nodes_data)}. "
            )

        # Nodal Masses (including beam and plate contributions)
        nodal_masses_tonne: Dict[int, np.ndarray] = {nid: np.zeros(6) for nid in model.points.keys()}

        # Add beam masses
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

        # Add plate masses
        plate_nodal_masses = self._calculate_plate_nodal_masses(model)
        for node_id, plate_mass in plate_nodal_masses.items():
            if node_id in nodal_masses_tonne:
                for i in range(3):  # Add to translational mass components
                    nodal_masses_tonne[node_id][i] += plate_mass

        # Apply nodal masses to OpenSees model
        for node_id, mass_values in nodal_masses_tonne.items():
            if np.sum(mass_values[:3]) > 1e-9:  # Only add mass if it's significant
                ops.mass(node_id, *mass_values)  # Use model node_id directly

        self.model_built = True
        self._model = model
        logger.info("OpenSees model setup complete.")
        if model.plates:
            logger.info(f"Model includes {len(model.plates)} steel plates with ShellMITC4 elements.")

    def apply_gravity_loads(self):
        """Applies gravity load as uniform beam loads in OpenSees."""
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup_opensees_model first.")

        ops.timeSeries("Linear", 1)  # Load Pattern TimeSeries Tag
        ops.pattern("Plain", 1, 1)  # Load Pattern Tag, TimeSeries Tag

        for beam_id, beam_obj in self._model.beams.items():
            section_props = self._model.sections.get(beam_obj.section)
            if not section_props:
                continue

            area_m2 = section_props.area
            w_N_per_m = area_m2 * RHO_STEEL_KG_M3 * G_M_S2
            w_N_per_mm = w_N_per_m * N_M_TO_N_MM

            if beam_id not in self.beam_local_coord_systems:
                logger.warning(f"Warning: Local axes not found for beam {beam_id}. Skipping gravity load.")
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
        logger.info("Gravity loads applied.")

    def run_static_analysis(self) -> bool:
        """Performs static analysis under the applied loads."""
        if not self.model_built:
            raise RuntimeError("Model not built.")
        if not self.loads_applied:
            logger.warning("Warning: No loads seem to have been applied before static analysis.")

        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")

        logger.info("Running static analysis...")
        result = ops.analyze(1)
        if result == 0:
            logger.info("Static analysis successful.")
            ops.reactions()  # Ensure reactions are calculated/updated
            self._static_analysis_success = True
        else:
            logger.error(f"Static analysis FAILED. Result code: {result}")
            self._static_analysis_success = False
        return self._static_analysis_success

    def run_modal_analysis(self, num_modes: int) -> Optional[List[float]]:
        """Performs modal analysis."""
        if not self.model_built:
            raise RuntimeError("Model not built.")

        logger.info(f"Running modal analysis for {num_modes} modes...")
        eigenvalues_omega_sq = ops.eigen(num_modes)
        if (
            eigenvalues_omega_sq is None
            or not isinstance(eigenvalues_omega_sq, list)
            or len(eigenvalues_omega_sq) == 0
        ):
            logger.error("Modal analysis failed or returned no eigenvalues.")
            return None

        logger.info("Modal analysis successful.")
        return eigenvalues_omega_sq

    def _process_modal_analysis_results(
        self, eigenvalues_rad2_s2: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Processes modal analysis results to return frequencies in Hz and periods in seconds."""
        if not eigenvalues_rad2_s2:
            return [], []

        frequencies_hz = [np.sqrt(val) / (2 * np.pi) for val in eigenvalues_rad2_s2 if val > 0]
        periods_s = [1 / f for f in frequencies_hz if f > 0]
        return frequencies_hz, periods_s

    def calculate_modal_participation_factors(self, numModes: int) -> List[float]:
        # 获取所有节点标签
        allNodes = ops.getNodeTags()

        # 计算总质量 (X和Y方向)
        totalMassX = sum(ops.nodeMass(node)[0] for node in allNodes)
        totalMassY = sum(ops.nodeMass(node)[1] for node in allNodes)
        totalMassZ = sum(ops.nodeMass(node)[2] for node in allNodes)

        print(f"Total Mass - X: {totalMassX}, Y: {totalMassY}")

        # 初始化结果存储
        modal_participation = np.zeros((numModes, 3))  # 存储X和Y方向的质量参与系数

        # 逐模态计算
        for mode in range(numModes):
            GammaX, GammaY, GammaZ = 0.0, 0.0, 0.0
            for node in allNodes:
                # 获取节点质量和振型分量 (X, Y, RZ)
                massX, massY, massZ, massRx, massRy, massRz = ops.nodeMass(node)
                phi = ops.nodeEigenvector(node, mode + 1)  # 注意：OpenSeesPy中模态编号从1开始
                phi_X, phi_Y, phi_Z = phi[0], phi[1], phi[2]  # 提取XYZ方向分量

                # 计算参与因子
                GammaX += massX * phi_X
                GammaY += massY * phi_Y
                GammaZ += massZ * phi_Z
            # 计算有效质量和参与系数
            effectiveMassX = GammaX**2
            effectiveMassY = GammaY**2
            effectiveMassZ = GammaZ**2

            modal_participation[mode, 0] = effectiveMassX / totalMassX * 100  # X方向百分比
            modal_participation[mode, 1] = effectiveMassY / totalMassY * 100  # Y方向百分比
            modal_participation[mode, 2] = effectiveMassZ / totalMassZ * 100  # Z方向百分比

        return modal_participation

    def get_node_displacements(self, query_node_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        Retrieves displacements for specified model node IDs (or all if None).
        Returns Dict[model_node_id, np.array_of_6_displacements_mm_rad]
        """
        if not self._static_analysis_success:
            logger.warning("Warning: Static analysis not run or failed.")
            return {}

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
                logger.error(f"Could not get displacement for OpenSees node tag {node_id}: {e}")
        return displacements

    def get_support_reactions(self) -> Dict[int, np.ndarray]:
        """
        Retrieves reactions for the defined support nodes.
        Returns Dict[model_node_id, np.array_of_6_reactions_N_Nmm]
        """
        reactions: Dict[int, np.ndarray] = {}
        if not self.support_node_ids:
            logger.warning("Warning: No support nodes defined/found to get reactions from.")
            return reactions

        for node_id in self.support_node_ids:  # Iterate original model IDs stored
            try:
                # Use model node_id directly as OpenSees tag
                react_N_Nmm = np.array(ops.nodeReaction(node_id))
                reactions[node_id] = react_N_Nmm
            except Exception as e:
                logger.error(f"Could not get reaction for OpenSees support node tag {node_id}: {e}")
        return reactions

    def save_results(self, data: Dict, filename: str, output_dir: str = "results"):
        """Saves dictionary data to a JSON file in the specified directory with improved serialization."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)

        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, "tolist"):  # Handle other numpy-like objects
                return obj.tolist()
            else:
                return obj

        serializable_data = convert_to_serializable(data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")

    def perform_full_analysis_and_save(
        self,
        model: StructuralModel,
        num_modal_modes: int = 5,
        output_dir: str = "opensees_results",
        internal_pressure_pa: float = 30000.0,
        include_pressure_loads: bool = True,
    ):
        """
        Convenience method to run the full sequence: setup, load, static & modal analysis, save results.
        Now includes steel plate analysis for pressurized buildings.

        Args:
            model: StructuralModel object
            num_modal_modes: Number of modal modes to analyze
            output_dir: Output directory for results
            internal_pressure_pa: Internal pressure for pressurized building in Pa
            include_pressure_loads: Whether to apply pressure loads to plates
        """
        logger.info("Starting OpenSees full analysis workflow with steel plate modeling...")
        self.setup_opensees_model(model)

        if self.model_built:
            # Apply loads based on analysis type
            if model.plates and include_pressure_loads:
                # For pressurized buildings: apply both gravity and pressure loads
                self.apply_pressure_loads(internal_pressure_pa)
            else:
                # Traditional analysis: gravity loads only
                self.apply_gravity_loads()

            static_success = self.run_static_analysis()
            if static_success:
                # Get traditional node displacements and reactions
                displacements = self.get_node_displacements()
                reactions = self.get_support_reactions()

                # Get plate-specific results
                plate_analysis_results = {}
                if model.plates:
                    logger.info("Analyzing steel plate deformation...")
                    plate_displacements = self.get_plate_displacements()
                    plate_stresses_results = self.get_plate_stresses()
                    plate_deformation_analysis = self.get_analysis_result(
                        plate_displacements, plate_stresses_results
                    )

                    plate_analysis_results = {
                        "plate_displacements": plate_displacements,
                        "plate_stresses": plate_stresses_results,
                        "plate_deformation_analysis": plate_deformation_analysis,
                        "analysis_parameters": {
                            "plate_thickness_mm": self.plate_thickness_mm,
                            "internal_pressure_pa": internal_pressure_pa if include_pressure_loads else 0,
                            "pressure_applied": include_pressure_loads,
                        },
                    }

                # Combine all static analysis results
                static_results_data = {
                    "displacements_mm_rad": displacements,
                    "support_reactions_N_Nmm": reactions,
                }

                # Add plate results if available
                if plate_analysis_results:
                    static_results_data["plate_analysis"] = plate_analysis_results

                self.save_results(static_results_data, "static_analysis_results.json", output_dir)

                # Print plate analysis summary
                if model.plates and plate_analysis_results:
                    deform_analysis = plate_analysis_results["plate_deformation_analysis"]
                    if "performance_indicators" in deform_analysis:
                        indicators = deform_analysis["performance_indicators"]
                        logger.info("\n=== Steel Plate Analysis Summary ===")
                        logger.info(
                            f"Total plates analyzed (displacement): {indicators.get('plates_analyzed_displacement', 0)}"
                        )
                        logger.info(
                            f"Total plates with stress results: {indicators.get('plates_analyzed_stress', 0)}"
                        )
                        logger.info(
                            f"Maximum plate nodal displacement: {indicators.get('max_plate_nodal_displacement_mm', 0):.3f} mm"
                        )
                        if indicators.get("max_overall_normal_stress_mpa") is not None:
                            logger.info(
                                f"Maximum overall absolute normal stress: {indicators.get('max_overall_normal_stress_mpa', 0):.3f} MPa (Plate ID: {indicators.get('plate_with_max_normal_stress', 'N/A')})"
                            )
                        logger.info(
                            f"Average plate center displacement: {indicators.get('average_plate_center_displacement_mm', 0):.3f} mm"
                        )
                        if include_pressure_loads:
                            logger.info(f"Internal pressure applied: {internal_pressure_pa} Pa")
                        logger.info("=====================================\n")

            # Modal analysis
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

        logger.info("OpenSees full analysis workflow finished.")

    def _create_plate_materials_and_sections(self, plate_thickness_mm: float = 10.0):
        """
        Create steel plate materials and sections for ShellMITC4 elements.
        Args:
            plate_thickness_mm: Plate thickness in mm (default 10mm for typical steel skin)
        """
        # Create isotropic elastic material for steel plates
        # nDMaterial ElasticIsotropic $matTag $E $nu <$rho>
        ops.nDMaterial(
            "ElasticIsotropic", self.plate_material_tag, E_STEEL_MPA, NU_STEEL, RHO_STEEL_TONNE_MM3
        )

        # Create elastic membrane plate section
        # section ElasticMembranePlateSection $secTag $E $nu $h $rho
        ops.section(
            "ElasticMembranePlateSection",
            self.plate_section_tag,
            E_STEEL_MPA,
            NU_STEEL,
            plate_thickness_mm,
            # RHO_STEEL_TONNE_MM3,
        )

        logger.info(
            f"Created plate material (tag: {self.plate_material_tag}) and section (tag: {self.plate_section_tag})"
        )
        logger.info(f"Plate thickness: {plate_thickness_mm} mm")

    def _add_plate_elements(self, model: StructuralModel):
        """
        Add ShellMITC4 plate elements to the OpenSees model.
        """
        if not model.plates:
            logger.warning("No plates found in the model. Skipping plate element creation.")
            return

        logger.info(f"Adding {len(model.plates)} plate elements...")

        for plate_id, plate_obj in model.plates.items():
            # Get the four node IDs in counter-clockwise order
            node_ids = plate_obj.nodes

            if len(node_ids) != 4:
                logger.warning(f"Warning: Plate {plate_id} does not have exactly 4 nodes. Skipping.")
                continue

            # Verify all nodes exist in the model
            missing_nodes = [nid for nid in node_ids if nid not in model.points]
            if missing_nodes:
                logger.warning(
                    f"Warning: Plate {plate_id} references missing nodes {missing_nodes}. Skipping."
                )
                continue

            try:
                # Create ShellMITC4 element
                # element ShellMITC4 $eleTag $iNode $jNode $kNode $lNode $secTag
                ops.element("ShellMITC4", plate_id, *node_ids, self.plate_section_tag)
                # print(f"Created plate element {plate_id} with nodes {node_ids}")

            except Exception as e:
                logger.error(f"Error creating plate element {plate_id}: {e}")

    def _calculate_plate_nodal_masses(self, model: StructuralModel) -> Dict[int, float]:
        """
        Calculate nodal masses from plates.
        Returns: Dict[node_id, additional_mass_tonne]
        """
        nodal_plate_masses: Dict[int, float] = {}

        if not model.plates:
            return nodal_plate_masses

        for plate_id, plate_obj in model.plates.items():
            # Calculate plate mass
            plate_area_m2 = plate_obj.area(model.points)
            plate_thickness_m = plate_obj.thickness
            plate_volume_m3 = plate_area_m2 * plate_thickness_m
            plate_mass_kg = plate_volume_m3 * RHO_STEEL_KG_M3
            plate_mass_tonne = plate_mass_kg / 1000.0

            # Distribute mass equally to 4 nodes
            mass_per_node = plate_mass_tonne / 4.0

            for node_id in plate_obj.nodes:
                if node_id in nodal_plate_masses:
                    nodal_plate_masses[node_id] += mass_per_node
                else:
                    nodal_plate_masses[node_id] = mass_per_node

        return nodal_plate_masses

    def apply_pressure_loads(self, internal_pressure_pa: float = 30000.0):
        """
        Applies internal pressure loads to steel plates by converting them to equivalent
        line loads on the boundary beams of each plate.

        Args:
            model: StructuralModel object containing plates
            internal_pressure_pa: Internal pressure in Pa (default 2000 Pa = 2 kPa for pressurized buildings)
        """
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup_opensees_model first.")

        if not self._model.plates:
            logger.warning("No plates found in the model. Skipping pressure load application.")
            return

        # Convert pressure from Pa to N/mm^2 (MPa)
        pressure_mpa = internal_pressure_pa / 1.0e6

        ops.timeSeries("Linear", 2)  # Different time series for pressure loads
        ops.pattern("Plain", 2, 2)  # Load Pattern Tag 2 for pressure

        pressure_loads_applied = 0
        total_equivalent_force = 0.0

        # Define pressure directions for each face (internal pressure acts outward)
        face_pressure_directions = {
            "top": np.array([0.0, 0.0, 1.0]),  # 向上
            "bottom": np.array([0.0, 0.0, -1.0]),  # 向下
            "left": np.array([0.0, -1.0, 0.0]),  # 向左 (Y负方向)
            "right": np.array([0.0, 1.0, 0.0]),  # 向右 (Y正方向)
            "front": np.array([-1.0, 0.0, 0.0]),  # 向前 (X正方向)
            "back": np.array([1.0, 0.0, 0.0]),  # 向后 (X负方向)
        }

        logger.info("Converting plate pressure loads to equivalent beam line loads...")

        for plate_id, plate_obj in self._model.plates.items():
            # if plate_obj.on_face != "back":
            #     continue
            try:
                # Calculate plate area
                plate_area_m2 = plate_obj.area(self._model.points)
                plate_area_mm2 = plate_area_m2 * M2_TO_MM2

                # Total force on the plate
                total_force_n = pressure_mpa * plate_area_mm2
                total_equivalent_force += total_force_n

                # Get pressure direction for this plate's face
                plate_face = plate_obj.on_face
                if plate_face not in face_pressure_directions:
                    logger.warning(
                        f"Warning: Unknown face '{plate_face}' for plate {plate_id}, using default upward direction"
                    )
                    pressure_direction = np.array([0.0, 0.0, 1.0])
                else:
                    pressure_direction = face_pressure_directions[plate_face]

                # print(f"  Plate {plate_id} on {plate_face} face: pressure direction {pressure_direction}")

                # Find beams that form the boundary of this plate
                plate_node_ids = set(plate_obj.nodes)
                boundary_beams = []

                for beam_id, beam_obj in self._model.beams.items():
                    beam_nodes = {beam_obj.start_node, beam_obj.end_node}
                    # Check if both beam nodes are in the plate nodes (i.e., beam is on plate boundary)
                    if beam_nodes.issubset(plate_node_ids):
                        # Calculate beam length
                        start_point = self._model.points[beam_obj.start_node]
                        end_point = self._model.points[beam_obj.end_node]
                        beam_length_m = start_point.distance_to(end_point)
                        beam_length_mm = beam_length_m * M_TO_MM

                        boundary_beams.append(
                            {"beam_id": beam_id, "beam_obj": beam_obj, "length_mm": beam_length_mm}
                        )

                if not boundary_beams:
                    logger.warning(f"Warning: No boundary beams found for plate {plate_id}")
                    continue

                # Calculate total perimeter length
                total_perimeter_mm = sum(beam["length_mm"] for beam in boundary_beams)

                if total_perimeter_mm <= 0:
                    logger.warning(f"Warning: Zero perimeter length for plate {plate_id}")
                    continue

                # Distribute the total force to boundary beams proportionally
                for beam_data in boundary_beams:
                    beam_id = beam_data["beam_id"]
                    beam_length_mm = beam_data["length_mm"]

                    # Force proportion based on beam length
                    force_proportion = beam_length_mm / total_perimeter_mm
                    beam_force_n = total_force_n * force_proportion

                    # Convert to distributed load (N/mm)
                    distributed_load_n_per_mm = beam_force_n / beam_length_mm

                    # Apply pressure load in the correct direction for this face
                    if beam_id not in self.beam_local_coord_systems:
                        logger.warning(
                            f"Warning: Local axes not found for beam {beam_id}. Skipping pressure load."
                        )
                        continue

                    x_loc, y_loc, z_loc = self.beam_local_coord_systems[beam_id]
                    R_glob_to_loc = np.array([x_loc, y_loc, z_loc])

                    # Pressure force in global coordinates (face-specific direction)
                    F_global_per_mm = pressure_direction * distributed_load_n_per_mm
                    F_local_components = R_glob_to_loc @ F_global_per_mm

                    load_val_local_x = F_local_components[0]
                    load_val_local_y = F_local_components[1]
                    load_val_local_z = F_local_components[2]

                    try:
                        # Apply uniform beam load in local coordinates
                        ops.eleLoad(
                            "-ele",
                            beam_id,
                            "-type",
                            "-beamUniform",
                            load_val_local_y,
                            load_val_local_z,
                            load_val_local_x,
                        )
                        pressure_loads_applied += 1

                    except Exception as e:
                        logger.warning(f"Warning: Could not apply pressure load to beam {beam_id}: {e}")

            except Exception as e:
                logger.error(f"Error processing pressure load for plate {plate_id}: {e}")

        self.pressure_load_applied = True
        logger.info(f"Pressure loads converted and applied to {pressure_loads_applied} boundary beams.")
        logger.info(f"Total equivalent force: {total_equivalent_force:.2f} N")
        logger.info(f"Pressure magnitude: {internal_pressure_pa} Pa ({pressure_mpa:.6f} MPa)")

    def apply_gravity_and_pressure_loads(self, internal_pressure_pa: float = 30000.0):
        """
        Applies both gravity loads (on beams) and internal pressure loads (on plates).
        This is the recommended method for pressurized building analysis.

        Args:
            model: StructuralModel object
            internal_pressure_pa: Internal pressure in Pa
        """
        # Apply gravity loads to beams
        self.apply_gravity_loads()

        # Apply pressure loads to plates
        if self._model.plates:
            self.apply_pressure_loads(internal_pressure_pa)

        logger.info("Combined gravity and pressure loads applied.")

    def get_plate_displacements(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Retrieves plate element nodal displacements and calculates plate-specific results.
        Returns: Dict[plate_id, {'nodes': node_ids, 'displacements': displacements_array}]
        """
        if not self._static_analysis_success:
            logger.warning("Warning: Static analysis not run or failed.")
            return {}

        if not self._model.plates:
            return {}

        plate_displacements = {}

        for plate_id, plate_obj in self._model.plates.items():
            try:
                # Get nodal displacements for the plate nodes
                node_disps = []
                for node_id in plate_obj.nodes:
                    try:
                        disp = np.array(ops.nodeDisp(node_id))
                        node_disps.append(disp)
                    except Exception as e:
                        logger.error(f"Could not get displacement for node {node_id}: {e}")
                        node_disps.append(np.zeros(6))

                plate_displacements[plate_id] = {
                    "nodes": plate_obj.nodes,
                    "displacements": np.array(node_disps),  # Shape: (4, 6) for 4 nodes, 6 DOFs each
                    "center_displacement": np.mean(
                        node_disps, axis=0
                    ),  # Average displacement at plate center
                }

            except Exception as e:
                logger.error(f"Error getting plate {plate_id} displacements: {e}")

        return plate_displacements

    def get_plate_stresses(self) -> Dict[int, np.ndarray]:
        """
        Retrieves plate element stresses and forces.
        Returns: Dict[plate_id, stress/force data]
        """
        if not self._static_analysis_success:
            logger.warning("Warning: Static analysis not run or failed.")
            return {}

        if not self._model.plates:
            return {}

        plate_stresses_results = {}  # Renamed to avoid conflict with the 'stresses' key

        for plate_model_id in self._model.plates.keys():  # plate_model_id is the key from model.plates
            try:
                # Get element stress/force response
                # For ShellMITC4, common response types are:
                # 'forces', 'stresses', 'material' responses

                # forces = ops.eleResponse(ele_tag, "forces")  # Use ele_tag
                # current_plate_data["forces"] = np.array(forces) if forces is not None else None

                stresses_response = ops.eleResponse(plate_model_id, "stresses")  # Use ele_tag
                if stresses_response is not None:
                    plate_stresses_results[plate_model_id] = np.array(stresses_response)
                else:
                    plate_stresses_results[plate_model_id] = None

            except Exception as e:
                logger.error(f"Error getting responses for plate element {plate_model_id}: {e}")
                plate_stresses_results[plate_model_id] = None

        return plate_stresses_results

    def get_beam_stresses(self) -> Dict[int, np.ndarray]:
        """
        Retrieves beam element stresses and forces.
        Returns: Dict[beam_id, stress/force data]
        """
        if not self._static_analysis_success:
            logger.warning("Warning: Static analysis not run or failed.")
            return {}

        beam_stresses_results = {}

        def calc_one_end(N, Mz, A, Iz, h):
            y_top = -h / 2  # 上缘
            y_bottom = h / 2  # 下缘

            # 轴向应力
            sigma_n = N / A

            # 弯曲应力 (上缘和下缘)
            sigma_m_top = (Mz * y_top) / Iz
            sigma_m_bottom = (Mz * y_bottom) / Iz

            # 总应力
            sigma_top = sigma_n + sigma_m_top
            sigma_bottom = sigma_n + sigma_m_bottom

            return sigma_top, sigma_bottom

        for beam_id, beam_obj in self._model.beams.items():
            forces = ops.eleResponse(beam_id, "force")
            if forces is None:
                beam_stresses_results[beam_id] = None
                continue

            N1, My1, Mz1 = forces[0], forces[4], forces[5]
            N2, My2, Mz2 = forces[6], forces[10], forces[11]

            section = self._model.sections[beam_obj.section]
            A = section.area * M2_TO_MM2
            Iz = section.Ix * M4_TO_MM4
            Iy = section.Iy * M4_TO_MM4
            hz = section.B if isinstance(section, RectangularTubeSection) else section.H
            hy = section.B if isinstance(section, RectangularTubeSection) else section.B

            sigma_top1, sigma_bottom1 = calc_one_end(N1, Mz1, A, Iz, hz)
            sigma_top2, sigma_bottom2 = calc_one_end(N2, Mz2, A, Iz, hz)
            sigma_top3, sigma_bottom3 = calc_one_end(N1, My1, A, Iy, hy)
            sigma_top4, sigma_bottom4 = calc_one_end(N2, My2, A, Iy, hy)

            beam_stresses_results[beam_id] = [
                sigma_top1,
                sigma_bottom1,
                sigma_top2,
                sigma_bottom2,
                sigma_top3,
                sigma_bottom3,
                sigma_top4,
                sigma_bottom4,
            ]

        return beam_stresses_results

    def get_analysis_result(
        self,
        plate_displacements: Dict[int, Dict[str, np.ndarray]],
        plate_stresses_results: Dict[int, Dict[str, Optional[np.ndarray]]],
    ) -> Dict[str, any]:
        """
        Analyzes plate deformation characteristics including maximum displacements,
        stress concentrations, and structural performance indicators.
        Now also calculates maximum normal stress from plate_stresses_results.
        """
        if not self._model.plates:  # Should not happen if plate_displacements/stresses are populated
            return {"message": "No plates in model to analyze"}

        # Initialize analysis_results structure
        analysis_results = {
            "total_plates": len(self._model.plates),
            "max_displacement": {"value": 0.0, "plate_id": None, "direction": None},
            "plate_summary": {},  # Will be populated with per-plate data
            "performance_indicators": {},
        }

        # Populate plate_summary with basic entries first
        for plate_id in self._model.plates.keys():
            analysis_results["plate_summary"][plate_id] = {}

        # Analyze displacements (using the provided plate_displacements)
        max_disp_magnitude = 0.0
        all_center_disp_magnitudes = []  # For calculating average and std dev

        for plate_id, disp_data in plate_displacements.items():
            if plate_id not in analysis_results["plate_summary"]:  # Should exist from above
                analysis_results["plate_summary"][plate_id] = {}

            displacements = disp_data.get("displacements")  # Nodal displacements (num_nodes, 6)
            center_disp_vector = disp_data.get("center_displacement")  # Average displacement (6,)

            if displacements is None or center_disp_vector is None:
                analysis_results["plate_summary"][plate_id].update(
                    {
                        "max_nodal_displacement_mm": None,
                        "center_displacement_mm": None,
                        "center_displacement_vector": None,
                    }
                )
                continue

            # Calculate maximum displacement magnitude for translational DOFs (0,1,2)
            trans_disps = displacements[:, :3]  # Translation displacements only
            max_nodal_trans_disp_for_plate = (
                np.max(np.linalg.norm(trans_disps, axis=1)) if trans_disps.size > 0 else 0.0
            )
            center_trans_disp_magnitude = (
                np.linalg.norm(center_disp_vector[:3]) if center_disp_vector.size >= 3 else 0.0
            )

            all_center_disp_magnitudes.append(center_trans_disp_magnitude)

            analysis_results["plate_summary"][plate_id].update(
                {
                    "max_nodal_displacement_mm": max_nodal_trans_disp_for_plate,
                    "center_displacement_mm": center_trans_disp_magnitude,
                    "center_displacement_vector": (
                        center_disp_vector[:3].tolist() if center_disp_vector.size >= 3 else None
                    ),
                }
            )

            if max_nodal_trans_disp_for_plate > max_disp_magnitude:
                max_disp_magnitude = max_nodal_trans_disp_for_plate
                analysis_results["max_displacement"] = {
                    "value": max_nodal_trans_disp_for_plate,
                    "plate_id": plate_id,
                    "direction": "translation_nodal",  # Clarify it's max nodal
                }

        # Displacement-based performance indicators
        if all_center_disp_magnitudes:
            analysis_results["performance_indicators"] = {
                "max_plate_nodal_displacement_mm": max_disp_magnitude,  # Renamed for clarity
                "average_plate_center_displacement_mm": np.mean(all_center_disp_magnitudes),
                "plate_center_displacement_std_mm": np.std(all_center_disp_magnitudes),
                "plates_analyzed_displacement": len(plate_displacements),
            }
        else:
            analysis_results["performance_indicators"] = {
                "max_plate_nodal_displacement_mm": 0.0,
                "average_plate_center_displacement_mm": 0.0,
                "plate_center_displacement_std_mm": 0.0,
                "plates_analyzed_displacement": 0,
            }

        # Analyze stresses for maximum normal stress
        max_overall_abs_normal_stress = -1.0
        plate_id_with_max_stress = None
        stresses_processed_count = 0

        for plate_model_id, stress_data in plate_stresses_results.items():
            stresses_array = stress_data

            if plate_model_id not in analysis_results["plate_summary"]:  # Should exist
                analysis_results["plate_summary"][plate_model_id] = {}

            if (
                stresses_array is None
                or not isinstance(stresses_array, np.ndarray)
                or stresses_array.size != 32
            ):
                analysis_results["plate_summary"][plate_model_id]["max_normal_stress_mpa"] = None
                continue

            stresses_processed_count += 1

            # Indices for s_xx: 0, 4, 8, 12, 16, 20, 24, 28 (at each of 8 stress points: 4 GPs x 2 surfaces)
            # Indices for s_yy: 1, 5, 9, 13, 17, 21, 25, 29
            # The 32 values are [GP1_Top(sxx,syy,sxy,szz), GP1_Bot(sxx,syy,sxy,szz), GP2_Top(...), ...]
            # So, s_xx are at indices 0, 4, 8, 12, 16, 20, 24, 28
            # And s_yy are at indices 1, 5, 9, 13, 17, 21, 25, 29

            s_xx_indices = np.array([i * 4 for i in range(8)])  # 0, 4, 8, ..., 28
            s_yy_indices = np.array([i * 4 + 1 for i in range(8)])  # 1, 5, 9, ..., 29

            current_plate_s_xx = stresses_array[s_xx_indices]
            current_plate_s_yy = stresses_array[s_yy_indices]

            current_plate_normal_stresses = np.concatenate((current_plate_s_xx, current_plate_s_yy))

            max_abs_stress_for_this_plate_val = 0.0
            if current_plate_normal_stresses.size > 0:
                max_abs_stress_for_this_plate_val = np.max(np.abs(current_plate_normal_stresses))

            analysis_results["plate_summary"][plate_model_id][
                "max_abs_normal_stress_mpa"
            ] = max_abs_stress_for_this_plate_val

            if max_abs_stress_for_this_plate_val > max_overall_abs_normal_stress:
                max_overall_abs_normal_stress = max_abs_stress_for_this_plate_val
                plate_id_with_max_stress = plate_model_id

        analysis_results["performance_indicators"]["plates_analyzed_stress"] = stresses_processed_count
        if stresses_processed_count > 0 and plate_id_with_max_stress is not None:
            analysis_results["performance_indicators"][
                "max_overall_normal_stress_mpa"
            ] = max_overall_abs_normal_stress
            analysis_results["performance_indicators"][
                "plate_with_max_normal_stress"
            ] = plate_id_with_max_stress
        else:
            analysis_results["performance_indicators"]["max_overall_normal_stress_mpa"] = None
            analysis_results["performance_indicators"]["plate_with_max_normal_stress"] = None

        return analysis_results

    def get_max_indicators(self) -> Dict[str, float]:
        """
        Retrieves the maximum values of the performance indicators.
        """
        displacements = self.get_node_displacements()
        max_disp_abs = 0.0
        max_disp_abs_idx = None
        max_node_disp = None
        for node_id, disp in displacements.items():
            disp_np = np.array(disp)
            current_max_disp_abs = np.max(np.abs(disp_np[:3]))
            if current_max_disp_abs > max_disp_abs:
                max_disp_abs = current_max_disp_abs
                max_disp_abs_idx = node_id
                max_node_disp = disp_np[:3]

        beams_stresses = self.get_beam_stresses()
        max_beam_stress_abs = 0.0
        max_beam_stress_abs_idx = None
        max_beam_stress = None
        for beam_id, stresses in beams_stresses.items():
            if stresses is None:
                continue
            current_max_beam_stress_abs = np.max(np.abs(stresses))
            if current_max_beam_stress_abs > max_beam_stress_abs:
                max_beam_stress_abs = current_max_beam_stress_abs
                max_beam_stress_abs_idx = beam_id
                max_beam_stress = stresses

        if self._model.plates:
            t = self.plate_thickness_mm
            stresses = self.get_plate_stresses()
            max_plate_stress_abs = 0.0
            max_plate_stress_abs_idx = None
            all_stresses = []
            for plate_id, stress_data in stresses.items():
                if stress_data is None:
                    continue

                # 按高斯点分组 (4 Gauss points × 8 components)
                stress_data = stress_data.reshape(4, 8)
                all_stresses.append(stress_data)

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
            print("max, min stress:")
            print((max_stress).tolist(), "MPa")
            print((min_stress).tolist(), "MPa")

        return {
            "max_abs_displacement": max_disp_abs,
            "max_abs_displacement_idx": max_disp_abs_idx,
            "max_abs_disp_node_coord": self._model.points[max_disp_abs_idx].coordinates(),
            "max_displacement_vector": max_node_disp,
            "max_abs_beam_stress": max_beam_stress_abs,
            "max_abs_beam_stress_idx": max_beam_stress_abs_idx,
            "max_beam_stress": max_beam_stress,
            "max_abs_plate_stress": max_plate_stress_abs,
            "max_abs_plate_stress_idx": max_plate_stress_abs_idx,
        }

    def visualize_model(self):
        b = 75
        h = 150
        fig_wi_he = (15, 15)
        ele_shapes = {1: ["circ", [h]], 2: ["rect", [b, h]], 3: ["I", [b, h, b / 10.0, h / 6.0]]}
        # opsv.plot_model(node_labels=0, element_labels=0, axis_off=1, fig_wi_he=fig_wi_he)
        # opsv.plot_extruded_shapes_3d(ele_shapes, fig_wi_he=fig_wi_he)
        opsv.plot_load(6)
        plt.show()


# Example Usage (can be in a main script or a test file)
if __name__ == "__main__":
    from src.utils.model_utils import create_structural_model, load_config

    logger.info("Testing OpenSeesAnalyzer...")
    config_path = "config.yaml"
    config = load_config(config_path)
    model = create_structural_model(config)
    print("结构总质量: ", model.calculate_mass() / 1000, "t")

    analyzer = OpenSeesAnalyzer()
    try:
        # analyzer.perform_full_analysis_and_save(
        #     model, num_modal_modes=6, plate_thickness_mm=5, output_dir="results/opensees_test_results_0idx"
        # )
        analyzer.setup_opensees_model(model)
        analyzer.apply_pressure_loads(internal_pressure_pa=30000.0)
        analyzer.run_static_analysis()
        max_indicators = analyzer.get_max_indicators()
        print("max_indicators: ", max_indicators)

        # max_nodal_idx
        max_nodal_idx = max_indicators["max_abs_displacement_idx"]
        # coordinates of the max nodal
        max_nodal_coord = model.points[max_nodal_idx].coordinates()
        print("max_nodal_coord: ", max_nodal_coord)

        # modal analysis
        num_modes = 12
        modal_results = analyzer.run_modal_analysis(num_modes=num_modes)
        frequencies_hz, periods_s = analyzer._process_modal_analysis_results(modal_results)
        print("frequencies_hz: ", frequencies_hz)
        print("periods_s: ", periods_s)
        # analyzer.visualize_model()
        modal_participation = analyzer.calculate_modal_participation_factors(numModes=num_modes)
        for i in range(num_modes):
            print(
                f"Mode {i+1}: X: {modal_participation[i, 0]:.2f}%, Y: {modal_participation[i, 1]:.2f}%, Z: {modal_participation[i, 2]:.2f}%"
            )

        disp_83 = analyzer.get_node_displacements([83])
        print("disp_83: ", disp_83)

        # analyzer.visualize_model()

    except Exception as e:
        logger.error(f"An error occurred during OpenSees analysis: {e}")
        import traceback

        traceback.print_exc()
