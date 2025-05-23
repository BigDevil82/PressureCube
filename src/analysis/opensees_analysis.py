import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import openseespy.opensees as ops

from src.structure.structural_model import Beam, Plate, StructuralModel  # For type hinting

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
        # Plate-related attributes
        self.PLATE_ELE_TAG_START = 10000
        self.plate_material_tag = 1000  # Starting tag for plate materials
        self.plate_section_tag = 2000  # Starting tag for plate sections
        self.pressure_load_applied = False

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

    def setup_opensees_model(self, model: StructuralModel, plate_thickness_mm: float = 10.0):
        """
        Creates the OpenSees model including nodes, materials, elements,
        boundary conditions, and nodal masses for modal analysis.
        Now includes steel plate modeling with ShellMITC4 elements.
        Units: N, mm, s, tonne.

        Args:
            model: StructuralModel object containing beams and plates
            plate_thickness_mm: Thickness of steel plates in mm (default 10mm)
        """
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)  # 3 dimensions, 6 DOFs per node

        # Create Nodes
        node_coords_mm_map: Dict[int, np.ndarray] = {}  # node_id -> coords_mm
        for node_id, point_obj in model.points.items():
            coords_mm = point_obj.coordinates() * M_TO_MM
            ops.node(node_id, *coords_mm)  # Use model node_id directly as OpenSees tag
            node_coords_mm_map[node_id] = coords_mm

        # Create plate materials and sections
        if model.plates:
            self._create_plate_materials_and_sections(plate_thickness_mm)

        # Geometric Transformations and Beam Elements
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

            # Assign a unique tag for each element's transformation
            current_geom_transf_tag = geom_transf_counter
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
            for node_id in potential_corners:
                if corner_nodes_found < 4 and node_id not in self.support_node_ids:
                    ops.fix(node_id, 1, 1, 1, 1, 1, 1)  # Fix all 6 DOFs, using model node_id
                    self.support_node_ids.append(node_id)
                    corner_nodes_found += 1

        if corner_nodes_found != 4 and len(model.points) > 0:  # Check if any nodes exist
            print(
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
        print("OpenSees model setup complete.")
        if model.plates:
            print(f"Model includes {len(model.plates)} steel plates with ShellMITC4 elements.")

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
        print(f"Results saved to {filepath}")

    def perform_full_analysis_and_save(
        self,
        model: StructuralModel,
        num_modal_modes: int = 5,
        output_dir: str = "opensees_results",
        plate_thickness_mm: float = 10.0,
        internal_pressure_pa: float = 2000.0,
        include_pressure_loads: bool = True,
    ):
        """
        Convenience method to run the full sequence: setup, load, static & modal analysis, save results.
        Now includes steel plate analysis for pressurized buildings.

        Args:
            model: StructuralModel object
            num_modal_modes: Number of modal modes to analyze
            output_dir: Output directory for results
            plate_thickness_mm: Steel plate thickness in mm
            internal_pressure_pa: Internal pressure for pressurized building in Pa
            include_pressure_loads: Whether to apply pressure loads to plates
        """
        print("Starting OpenSees full analysis workflow with steel plate modeling...")
        self.setup_opensees_model(model, plate_thickness_mm)

        if self.model_built:
            # Apply loads based on analysis type
            if model.plates and include_pressure_loads:
                # For pressurized buildings: apply both gravity and pressure loads
                self.apply_gravity_and_pressure_loads(model, internal_pressure_pa)
            else:
                # Traditional analysis: gravity loads only
                self.apply_gravity_loads(model)

            static_success = self.run_static_analysis()
            if static_success:
                # Get traditional node displacements and reactions
                displacements = self.get_node_displacements()
                reactions = self.get_support_reactions()

                # Get plate-specific results
                plate_analysis_results = {}
                if model.plates:
                    print("Analyzing steel plate deformation...")
                    plate_displacements = self.get_plate_displacements(model)
                    plate_stresses = self.get_plate_stresses(model)
                    plate_deformation_analysis = self.analyze_plate_deformation(model)

                    plate_analysis_results = {
                        "plate_displacements": plate_displacements,
                        "plate_stresses": plate_stresses,
                        "plate_deformation_analysis": plate_deformation_analysis,
                        "analysis_parameters": {
                            "plate_thickness_mm": plate_thickness_mm,
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
                        print("\n=== Steel Plate Analysis Summary ===")
                        print(f"Total plates analyzed: {indicators.get('plates_analyzed', 0)}")
                        print(
                            f"Maximum plate displacement: {indicators.get('max_plate_displacement_mm', 0):.3f} mm"
                        )
                        print(
                            f"Average plate displacement: {indicators.get('average_plate_displacement_mm', 0):.3f} mm"
                        )
                        if include_pressure_loads:
                            print(f"Internal pressure applied: {internal_pressure_pa} Pa")
                        print("=====================================\n")

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

        print("OpenSees full analysis workflow finished.")

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
            RHO_STEEL_TONNE_MM3,
        )

        print(
            f"Created plate material (tag: {self.plate_material_tag}) and section (tag: {self.plate_section_tag})"
        )
        print(f"Plate thickness: {plate_thickness_mm} mm")

    def _add_plate_elements(self, model: StructuralModel):
        """
        Add ShellMITC4 plate elements to the OpenSees model.
        """
        if not model.plates:
            print("No plates found in the model. Skipping plate element creation.")
            return

        print(f"Adding {len(model.plates)} plate elements...")

        for plate_id, plate_obj in model.plates.items():
            # Get the four node IDs in counter-clockwise order
            node_ids = plate_obj.nodes

            if len(node_ids) != 4:
                print(f"Warning: Plate {plate_id} does not have exactly 4 nodes. Skipping.")
                continue

            # Verify all nodes exist in the model
            missing_nodes = [nid for nid in node_ids if nid not in model.points]
            if missing_nodes:
                print(f"Warning: Plate {plate_id} references missing nodes {missing_nodes}. Skipping.")
                continue

            try:
                # Create ShellMITC4 element
                # element ShellMITC4 $eleTag $iNode $jNode $kNode $lNode $secTag
                ele_id = self.PLATE_ELE_TAG_START + plate_id
                ops.element("ShellMITC4", ele_id, *node_ids, self.plate_section_tag)
                # print(f"Created plate element {plate_id} with nodes {node_ids}")

            except Exception as e:
                print(f"Error creating plate element {plate_id}: {e}")

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

    def apply_pressure_loads(self, model: StructuralModel, internal_pressure_pa: float = 2000.0):
        """
        Applies internal pressure loads to steel plates by converting them to equivalent
        line loads on the boundary beams of each plate.

        Args:
            model: StructuralModel object containing plates
            internal_pressure_pa: Internal pressure in Pa (default 2000 Pa = 2 kPa for pressurized buildings)
        """
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup_opensees_model first.")

        if not model.plates:
            print("No plates found in the model. Skipping pressure load application.")
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
            "left": np.array([0.0, -1.0, 0.0]),  # 向前 (Y负方向)
            "right": np.array([0.0, 1.0, 0.0]),  # 向后 (Y正方向)
            "front": np.array([1.0, 0.0, 0.0]),  # 向左 (X负方向)
            "back": np.array([-1.0, 0.0, 0.0]),  # 向右 (X正方向)
        }

        print(f"Converting plate pressure loads to equivalent beam line loads...")

        for plate_id, plate_obj in model.plates.items():
            try:
                # Calculate plate area
                plate_area_m2 = plate_obj.area(model.points)
                plate_area_mm2 = plate_area_m2 * M2_TO_MM2

                # Total force on the plate
                total_force_n = pressure_mpa * plate_area_mm2
                total_equivalent_force += total_force_n

                # Get pressure direction for this plate's face
                plate_face = plate_obj.on_face
                if plate_face not in face_pressure_directions:
                    print(
                        f"Warning: Unknown face '{plate_face}' for plate {plate_id}, using default upward direction"
                    )
                    pressure_direction = np.array([0.0, 0.0, 1.0])
                else:
                    pressure_direction = face_pressure_directions[plate_face]

                # print(f"  Plate {plate_id} on {plate_face} face: pressure direction {pressure_direction}")

                # Find beams that form the boundary of this plate
                plate_node_ids = set(plate_obj.nodes)
                boundary_beams = []

                for beam_id, beam_obj in model.beams.items():
                    beam_nodes = {beam_obj.start_node, beam_obj.end_node}
                    # Check if both beam nodes are in the plate nodes (i.e., beam is on plate boundary)
                    if beam_nodes.issubset(plate_node_ids):
                        # Calculate beam length
                        start_point = model.points[beam_obj.start_node]
                        end_point = model.points[beam_obj.end_node]
                        beam_length_m = start_point.distance_to(end_point)
                        beam_length_mm = beam_length_m * M_TO_MM

                        boundary_beams.append(
                            {"beam_id": beam_id, "beam_obj": beam_obj, "length_mm": beam_length_mm}
                        )

                if not boundary_beams:
                    print(f"Warning: No boundary beams found for plate {plate_id}")
                    continue

                # Calculate total perimeter length
                total_perimeter_mm = sum(beam["length_mm"] for beam in boundary_beams)

                if total_perimeter_mm <= 0:
                    print(f"Warning: Zero perimeter length for plate {plate_id}")
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
                        print(f"Warning: Local axes not found for beam {beam_id}. Skipping pressure load.")
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
                        print(f"Warning: Could not apply pressure load to beam {beam_id}: {e}")

            except Exception as e:
                print(f"Error processing pressure load for plate {plate_id}: {e}")

        self.pressure_load_applied = True
        print(f"Pressure loads converted and applied to {pressure_loads_applied} boundary beams.")
        print(f"Total equivalent force: {total_equivalent_force:.2f} N")
        print(f"Pressure magnitude: {internal_pressure_pa} Pa ({pressure_mpa:.6f} MPa)")

    def apply_gravity_and_pressure_loads(self, model: StructuralModel, internal_pressure_pa: float = 2000.0):
        """
        Applies both gravity loads (on beams) and internal pressure loads (on plates).
        This is the recommended method for pressurized building analysis.

        Args:
            model: StructuralModel object
            internal_pressure_pa: Internal pressure in Pa
        """
        # Apply gravity loads to beams
        self.apply_gravity_loads(model)

        # Apply pressure loads to plates
        if model.plates:
            self.apply_pressure_loads(model, internal_pressure_pa)

        print("Combined gravity and pressure loads applied.")

    def get_plate_displacements(self, model: StructuralModel) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Retrieves plate element nodal displacements and calculates plate-specific results.
        Returns: Dict[plate_id, {'nodes': node_ids, 'displacements': displacements_array}]
        """
        if not model.plates:
            return {}

        plate_displacements = {}

        for plate_id, plate_obj in model.plates.items():
            try:
                # Get nodal displacements for the plate nodes
                node_disps = []
                for node_id in plate_obj.nodes:
                    try:
                        disp = np.array(ops.nodeDisp(node_id))
                        node_disps.append(disp)
                    except Exception as e:
                        print(f"Could not get displacement for node {node_id}: {e}")
                        node_disps.append(np.zeros(6))

                plate_displacements[plate_id] = {
                    "nodes": plate_obj.nodes,
                    "displacements": np.array(node_disps),  # Shape: (4, 6) for 4 nodes, 6 DOFs each
                    "center_displacement": np.mean(
                        node_disps, axis=0
                    ),  # Average displacement at plate center
                }

            except Exception as e:
                print(f"Error getting plate {plate_id} displacements: {e}")

        return plate_displacements

    def get_plate_stresses(self, model: StructuralModel) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Retrieves plate element stresses and forces.
        Returns: Dict[plate_id, stress/force data]
        """
        if not model.plates:
            return {}

        plate_stresses = {}

        for plate_id, plate_obj in model.plates.items():
            try:
                # Get element stress/force response
                # For ShellMITC4, common response types are:
                # 'forces', 'stresses', 'material' responses

                # Try to get element forces (membrane forces and moments)
                try:
                    forces = ops.eleResponse(plate_id, "forces")
                    plate_stresses[plate_id] = {"forces": np.array(forces) if forces else None}
                except:
                    plate_stresses[plate_id] = {"forces": None}

                # Try to get element stresses
                try:
                    stresses = ops.eleResponse(plate_id, "stresses")
                    if stresses:
                        plate_stresses[plate_id]["stresses"] = np.array(stresses)
                    else:
                        plate_stresses[plate_id]["stresses"] = None
                except:
                    plate_stresses[plate_id]["stresses"] = None

            except Exception as e:
                print(f"Error getting plate {plate_id} stresses: {e}")
                plate_stresses[plate_id] = {"forces": None, "stresses": None}

        return plate_stresses

    def analyze_plate_deformation(self, model: StructuralModel) -> Dict[str, any]:
        """
        Analyzes plate deformation characteristics including maximum displacements,
        stress concentrations, and structural performance indicators.
        """
        if not model.plates:
            return {"message": "No plates in model"}

        plate_displacements = self.get_plate_displacements(model)
        plate_stresses = self.get_plate_stresses(model)

        analysis_results = {
            "total_plates": len(model.plates),
            "max_displacement": {"value": 0.0, "plate_id": None, "direction": None},
            "plate_summary": {},
            "performance_indicators": {},
        }

        # Analyze displacements
        max_disp_magnitude = 0.0
        for plate_id, plate_data in plate_displacements.items():
            displacements = plate_data["displacements"]
            center_disp = plate_data["center_displacement"]

            # Calculate maximum displacement magnitude for translational DOFs (0,1,2)
            trans_disps = displacements[:, :3]  # Translation displacements only
            max_trans_disp = np.max(np.abs(trans_disps))
            center_trans_disp = np.linalg.norm(center_disp[:3])

            analysis_results["plate_summary"][plate_id] = {
                "max_nodal_displacement_mm": max_trans_disp,
                "center_displacement_mm": center_trans_disp,
                "center_displacement_vector": center_disp[:3].tolist(),
            }

            if max_trans_disp > max_disp_magnitude:
                max_disp_magnitude = max_trans_disp
                analysis_results["max_displacement"] = {
                    "value": max_trans_disp,
                    "plate_id": plate_id,
                    "direction": "translation",
                }

        # Calculate performance indicators
        if plate_displacements:
            all_center_disps = [
                data["center_displacement_mm"] for data in analysis_results["plate_summary"].values()
            ]
            analysis_results["performance_indicators"] = {
                "max_plate_displacement_mm": max_disp_magnitude,
                "average_plate_displacement_mm": np.mean(all_center_disps),
                "displacement_std_mm": np.std(all_center_disps),
                "plates_analyzed": len(plate_displacements),
            }

        return analysis_results


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
