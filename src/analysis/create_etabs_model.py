import json
import os
import time
from typing import Dict, List, Tuple

import comtypes.client
import numpy as np

from src.analysis.etabs_util import (  # Using specific imports for clarity
    MatType,
    ShellType,
    SlabType,
    Units_sys,
    create_csi_object,
    define_Conc_Mat,
    define_slab_sec,
    define_Steel_Mat,
    op_result,
)
from src.analysis.unit import GPa, M, MPa, N, kg, mm  # For constants and conversions
from src.structure.section_library import CSection, HSection, RectangularTubeSection

# Assuming these paths are correct relative to the project root for the example
from src.structure.structural_model import Beam, BeamType, Plate, Point, StructuralModel, StructuralParameters

# Constants for material properties
STEEL_E_VAL = 200 * 1e3  # N/mm^2 (MPa) (Converted from 200 GPa)
STEEL_MIU = 0.3
STEEL_THERMAL_EXP = 1.2e-5  # /C
STEEL_FY_VAL = 345  # N/mm^2 (MPa)
STEEL_FU_VAL = 470  # N/mm^2 (MPa)
STEEL_FYE_VAL = 380  # N/mm^2 (MPa) - Expected yield strength (for SetOSteel)
STEEL_FUE_VAL = 520  # N/mm^2 (MPa) - Expected tensile strength (for SetOSteel)

CONC_E_VAL = 30 * 1e3  # N/mm^2 (MPa) (Converted from 30 GPa)
CONC_MIU = 0.2
CONC_THERMAL_EXP = 1.0e-5  # /C

# Material Names
STEEL_MAT_NAME = "Q345"
CONCRETE_MAT_NAME = "C30"


def calculate_plate_nodal_masses(model: StructuralModel) -> Dict[int, float]:
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
        plate_mass_kg = plate_volume_m3 * 7850.0
        plate_mass_tonne = plate_mass_kg / 1000.0

        # Distribute mass equally to 4 nodes
        mass_per_node = plate_mass_tonne / 4.0

        for node_id in plate_obj.nodes:
            if node_id in nodal_plate_masses:
                nodal_plate_masses[node_id] += mass_per_node
            else:
                nodal_plate_masses[node_id] = mass_per_node

    return nodal_plate_masses


def calculate_nodal_mass(model: StructuralModel):
    """
    Calculate the nodal mass of the structural model.
    """
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
        length_mm = length_m * 1000

        area_mm2 = section_props.area * 1e6
        beam_mass_tonne = area_mm2 * length_mm * 7850.0 * 1e-12
        mass_per_node = beam_mass_tonne / 2.0

        start_node_id = beam_obj.start_node
        end_node_id = beam_obj.end_node

        for i in range(3):  # Add to translational mass components
            nodal_masses_tonne[start_node_id][i] += mass_per_node
            nodal_masses_tonne[end_node_id][i] += mass_per_node

    # Add plate masses
    plate_nodal_masses = calculate_plate_nodal_masses(model)
    for node_id, plate_mass in plate_nodal_masses.items():
        if node_id in nodal_masses_tonne:
            for i in range(3):  # Add to translational mass components
                nodal_masses_tonne[node_id][i] += plate_mass

    return nodal_masses_tonne


def find_bounary_corners(model: StructuralModel):
    """
    Find the boundary corners of the structural model.
    """
    # Boundary Conditions: Fix bottom 4 outermost nodes
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")
    bottom_nodes_data: List[Tuple[int, np.ndarray]] = []  # (node_id, coords_mm)

    for node_id, point_obj in model.points.items():  # Using model.points
        if np.isclose(point_obj.z, 0.0):  # Bottom nodes
            coords_mm = point_obj.coordinates()
            bottom_nodes_data.append((node_id, coords_mm))
            min_x = min(min_x, coords_mm[0])
            max_x = max(max_x, coords_mm[0])
            min_y = min(min_y, coords_mm[1])
            max_y = max(max_y, coords_mm[1])

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

    return potential_corners


def build_etabs_model(
    structural_model: StructuralModel,
    attach_mode=True,
    program_path="",
    program_target="ETABS",
    create_new_model=True,
):
    """
    Builds an ETABS model based on the provided StructuralModel instance.

    Args:
        structural_model: An instance of StructuralModel containing geometry and section info.
        attach_mode: If True, attempts to attach to a running ETABS instance.
        program_path: Path to ETABS.exe if creating a new instance and it's not in default location.
        program_target: "ETABS" or "SAP2000"
        create_new_model: If True, create a new model. If False, open the model in the current ETABS instance.

    Returns:
        Tuple (SapModel, myETABSObject, created_new_instance) if successful, otherwise (None, None, False).
    """
    # structural_model.plates = {}
    nodal_masses_tonne = calculate_nodal_mass(structural_model)
    support_node_ids = find_bounary_corners(structural_model)

    # 1. Initialize ETABS
    SapModel, myETABSObject, created_new_instance = create_csi_object(
        program_target=program_target, attach_mode=attach_mode, specific_program_path=program_path
    )

    if SapModel is None:
        print("Failed to get ETABS SapModel object.")
        return None, None, False

    # create a new model
    if create_new_model:
        ret = SapModel.InitializeNewModel()
        ret = SapModel.File.NewBlank()
        op_result(ret, f"Create a new model")

    # 2. Set Units
    # Ensure ETABS model uses N, mm, C
    ret = SapModel.SetPresentUnits(Units_sys.N_mm_C.value)
    op_result(ret, f"SetPresentUnits to {Units_sys.N_mm_C.name}")
    if ret != 0:
        print("Failed to set units. Aborting model creation.")
        if created_new_instance and myETABSObject:
            try:
                myETABSObject.ApplicationExit(False)
            except Exception as e_exit:
                print(f"Error trying to exit ETABS: {e_exit}")
        return None, None, created_new_instance  # Indicate failure

    # 3. Define Materials
    # Define a primary steel material.
    # Values for E, fy, fu should be in the current ETABS units (N/mm^2 or MPa)
    define_Steel_Mat(
        SapModel,
        STEEL_MAT_NAME,
        STEEL_E_VAL,
        STEEL_MIU,
        STEEL_THERMAL_EXP,
        fy=STEEL_FY_VAL,
        fu=STEEL_FU_VAL,
        fye=STEEL_FYE_VAL,
        fue=STEEL_FUE_VAL,
    )

    # 4. Define Sections
    # Frame Sections (Beams)
    # Assumes structural_model.sections contains SteelSection objects with properties in mm-based units.
    defined_beam_sections = {}  # To track successfully defined sections
    print("\n--- Defining Frame Sections ---")
    for sec_key, sec_props in structural_model.sections.items():
        # sec_props is a SteelSection object. Its name attribute is used as ETABS section name.
        # Properties like area, Ix, Iy, Iz are assumed to be in mm^2, mm^4 respectively.
        etabs_sec_name = sec_props.name
        if not etabs_sec_name:  # Skip if section name is empty
            print(f"Warning: Section with key '{sec_key}' has an empty name. Skipping.")
            continue

        if etabs_sec_name not in defined_beam_sections:
            print(f"Defining frame section: {etabs_sec_name} with material {STEEL_MAT_NAME}")

            # 根据截面类型使用不同的API函数
            ret_code = -1  # 默认错误码

            # 判断截面类型，使用对应的截面定义API
            if isinstance(sec_props, HSection):  # H型钢
                # SetISection参数：(Name, MatProp, T3, T2, Tf, Tw, T2b, Tfb, Color, Notes, GUID)
                # T3 - 截面总高度H
                # T2 - 上翼缘宽度B
                # Tf - 上翼缘厚度t_flange
                # Tw - 腹板厚度t_web
                # T2b - 下翼缘宽度(通常和上翼缘相同)
                # Tfb - 下翼缘厚度(通常和上翼缘相同)
                T3 = sec_props.H * M  # 转换为mm (ETABS单位)
                T2 = sec_props.B * M
                Tf = sec_props.t_flange * M
                Tw = sec_props.t_web * M
                T2b = T2  # 默认下翼缘宽度等于上翼缘
                Tfb = Tf  # 默认下翼缘厚度等于上翼缘

                print(f"  H型钢参数: 高H={T3:.1f}mm, 宽B={T2:.1f}mm, 翼缘厚={Tf:.1f}mm, 腹板厚={Tw:.1f}mm")
                ret_code = SapModel.PropFrame.SetISection(
                    etabs_sec_name, STEEL_MAT_NAME, T3, T2, Tf, Tw, T2b, Tfb
                )

            elif isinstance(sec_props, RectangularTubeSection):  # 方钢管
                # SetTube参数：(Name, MatProp, T3, T2, Tf, Tw, Color, Notes, GUID)
                # T3 - 截面高度B (对于方形，高=宽)
                # T2 - 截面宽度B (对于方形，高=宽)
                # Tf = Tw - 壁厚t (方钢管四边壁厚相同)
                T3 = sec_props.B * M
                T2 = sec_props.B * M
                Tf = sec_props.t * M
                Tw = sec_props.t * M

                print(f"  方钢管参数: 边长={T3:.1f}mm, 壁厚={Tf:.1f}mm")
                ret_code = SapModel.PropFrame.SetTube(etabs_sec_name, STEEL_MAT_NAME, T3, T2, Tf, Tw)

            elif isinstance(sec_props, CSection):  # 槽钢
                # SetChannel参数：(Name, MatProp, T3, T2, Tf, Tw, Color, Notes, GUID)
                # T3 - 截面高度H
                # T2 - 翼缘宽度B
                # Tf - 翼缘厚度(通常近似为壁厚)
                # Tw - 腹板厚度(壁厚)
                T3 = sec_props.H * M  # 转换为mm
                T2 = sec_props.B * M
                Tf = sec_props.t_web * M  # 假设翼缘厚度等于腹板厚度
                Tw = sec_props.t_web * M

                print(f"  槽钢参数: 高H={T3:.1f}mm, 宽B={T2:.1f}mm, 厚度={Tw:.1f}mm")
                ret_code = SapModel.PropFrame.SetChannel(etabs_sec_name, STEEL_MAT_NAME, T3, T2, Tf, Tw)

            # 如果未能识别类型，则使用通用截面定义(不太可能发生，仅作为备选)
            else:
                print(f"  未识别的截面类型: {type(sec_props).__name__}, 尝试使用通用截面定义")
                # 从属性中获取基本参数
                T3_val = 0.0
                T2_val = 0.0
                if hasattr(sec_props, "H") and hasattr(sec_props, "B"):
                    T3_val = sec_props.H * M
                    T2_val = sec_props.B * M

                area_val = sec_props.area * (M * M)
                as2_val = area_val * 0.8  # 估计值
                as3_val = area_val * 0.8  # 估计值
                torsion_val = sec_props.Iz * (M**4)
                i22_val = sec_props.Iy * (M**4)
                i33_val = sec_props.Ix * (M**4)
                s22_val = sec_props.Zy * (M**3)
                s33_val = sec_props.Zx * (M**3)
                z22_val = 1.15 * s22_val  # 估计值：塑性模量≈1.15*弹性模量
                z33_val = 1.15 * s33_val  # 估计值
                r22_val = (i22_val / area_val) ** 0.5 if i22_val and area_val > 1e-9 else 0.0
                r33_val = (i33_val / area_val) ** 0.5 if i33_val and area_val > 1e-9 else 0.0

                ret_code = SapModel.PropFrame.SetGeneral(
                    etabs_sec_name,
                    STEEL_MAT_NAME,
                    T3_val,
                    T2_val,
                    area_val,
                    as2_val,
                    as3_val,
                    torsion_val,
                    i22_val,
                    i33_val,
                    s22_val,
                    s33_val,
                    z22_val,
                    z33_val,
                    r22_val,
                    r33_val,
                )

            op_result(ret_code, f"Define Frame Section {etabs_sec_name}")
            if ret_code == 0:
                defined_beam_sections[etabs_sec_name] = True
            else:
                print(f"ERROR: Failed to define section {etabs_sec_name}. Error code: {ret_code}")

    # Shell Sections (Plates)
    # Plate thickness in structural_model is assumed to be in meters. Convert to mm.
    defined_shell_sections = {}
    print("\n--- Defining Shell Sections ---")
    for plate_id, plate_obj in structural_model.plates.items():
        thickness_m = plate_obj.thickness
        thickness_mm_val = thickness_m * M  # Convert meters to mm using unit.py constant

        shell_sec_name = f"Shell_T{thickness_mm_val:.2f}".replace(".", "_")  # e.g., Shell_T10_00 for 10mm
        if shell_sec_name not in defined_shell_sections:
            print(f"Defining shell section: {shell_sec_name} with thickness {thickness_mm_val:.2f} mm")
            ret_slab_sec = define_slab_sec(
                SapModel,
                shell_sec_name,
                SlabType.Slab.value,
                ShellType.ShellThin.value,
                STEEL_MAT_NAME,
                thickness_mm_val,
                program_target=program_target,
            )
            if ret_slab_sec == 0:
                defined_shell_sections[shell_sec_name] = True

    # 5. Create Nodes (Points)
    # Point coordinates in structural_model are assumed to be in meters. Convert to mm.
    node_sm_id_to_etabs_name = {}  # Map structural_model point ID to ETABS assigned node name
    print("\n--- Creating Nodes ---")
    for pt_id, pt_obj in structural_model.points.items():
        sm_node_name_req = str(pt_id)  # Requested name (using structural model ID)
        x_m = pt_obj.x
        y_m = pt_obj.y
        z_m = pt_obj.z

        # Convert coordinates from meters to mm
        x_mm_val = x_m * M
        y_mm_val = y_m * M
        z_mm_val = z_m * M

        # AddCartesian returns a tuple: (ActualNameAssignedByETABS, ReturnCode)
        etabs_assigned_node_name, ret_code = SapModel.PointObj.AddCartesian(
            x_mm_val, y_mm_val, z_mm_val, UserName=sm_node_name_req
        )
        op_result(
            ret_code,
            f"Add Point (req: {sm_node_name_req}, ETABS: {etabs_assigned_node_name}) at ({x_mm_val:.2f}, {y_mm_val:.2f}, {z_mm_val:.2f}) mm",
        )
        if ret_code == 0:
            node_sm_id_to_etabs_name[pt_id] = etabs_assigned_node_name
            nodal_mass = nodal_masses_tonne[pt_id].tolist()
            SapModel.PointObj.SetMass(etabs_assigned_node_name, nodal_mass)
            op_result(ret_code, f"Set mass for point {etabs_assigned_node_name} to {nodal_mass}")
            if pt_id in support_node_ids:
                SapModel.PointObj.SetRestraint(etabs_assigned_node_name, [1, 1, 1, 1, 1, 1])
                op_result(ret_code, f"Set restraint for point {etabs_assigned_node_name}")
            else:
                SapModel.PointObj.SetRestraint(etabs_assigned_node_name, [0, 0, 0, 0, 0, 0])
                op_result(ret_code, f"Set restraint for point {etabs_assigned_node_name}")
        else:
            print(f"ERROR: Failed to add point requested as {sm_node_name_req}.")

    # 6. Create Beams (Frame Objects)
    print("\n--- Creating Beams ---")
    for beam_id, beam_obj in structural_model.beams.items():
        sm_beam_name_req = str(beam_id)  # Requested name
        start_node_sm_id = beam_obj.start_node
        end_node_sm_id = beam_obj.end_node

        if start_node_sm_id not in node_sm_id_to_etabs_name or end_node_sm_id not in node_sm_id_to_etabs_name:
            print(
                f"Warning: Skipping beam {sm_beam_name_req}. Start or end node not found in ETABS mapping (IDs: {start_node_sm_id}, {end_node_sm_id})."
            )
            continue

        etabs_start_node_name = node_sm_id_to_etabs_name[start_node_sm_id]
        etabs_end_node_name = node_sm_id_to_etabs_name[end_node_sm_id]

        # Get the ETABS section name for this beam
        sm_section_key = beam_obj.section  # e.g., "FRAME", "MAIN"
        if sm_section_key not in structural_model.sections:
            print(
                f"Warning: Skipping beam {sm_beam_name_req}. Section key '{sm_section_key}' not found in structural_model.sections."
            )
            continue

        steel_section_obj = structural_model.sections[sm_section_key]
        etabs_section_name_for_beam = steel_section_obj.name

        if not etabs_section_name_for_beam or etabs_section_name_for_beam not in defined_beam_sections:
            print(
                f"Warning: Skipping beam {sm_beam_name_req}. Section '{etabs_section_name_for_beam}' was not defined or failed definition in ETABS."
            )
            continue

        # AddByPoint returns a tuple: (ActualNameAssignedByETABS, ReturnCode)
        etabs_assigned_beam_name, ret_code = SapModel.FrameObj.AddByPoint(
            etabs_start_node_name,
            etabs_end_node_name,
            PropName=etabs_section_name_for_beam,
            UserName=sm_beam_name_req,
        )
        op_result(
            ret_code,
            f"Add Beam (req: {sm_beam_name_req}, ETABS: {etabs_assigned_beam_name}) from '{etabs_start_node_name}' to '{etabs_end_node_name}' with section '{etabs_section_name_for_beam}'",
        )

        # set local axis angle
        if beam_obj.on_face not in ["top", "bottom"]:
            direction = beam_obj.direction(structural_model.points)
            if beam_obj.on_face in ["front", "back"] and direction[2] > 0:
                continue  # 前后面的竖向梁不需要旋转
            ret_code = SapModel.FrameObj.SetLocalAxes(etabs_assigned_beam_name, 90, 0)
            op_result(ret_code, f"Set local axis angle for beam {etabs_assigned_beam_name}")

        # 确保已添加Pressure荷载工况（如果在程序其他部分已定义可跳过这步）

    # 添加Pressure荷载工况
    SapModel.LoadPatterns.Add("Pressure", 1)  # 1对应DEAD荷载工况类型
    print("Created Pressure load pattern.")

    # 7. Create Plates (Area Objects / Shells)
    print("\n--- Creating Plates (Shells) and applying loads ---")
    for plate_id, plate_obj in structural_model.plates.items():
        sm_plate_name_req = str(plate_id)  # Requested name

        sm_node_ids_for_plate = plate_obj.nodes
        if len(sm_node_ids_for_plate) < 3:
            print(
                f"Warning: Skipping plate {sm_plate_name_req}. It has less than 3 nodes ({len(sm_node_ids_for_plate)})."
            )
            continue

        etabs_node_names_for_plate_list = []
        all_nodes_found = True
        for sm_node_id in sm_node_ids_for_plate:
            if sm_node_id not in node_sm_id_to_etabs_name:
                print(
                    f"Warning: Skipping plate {sm_plate_name_req}. Node ID {sm_node_id} not found in ETABS mapping."
                )
                all_nodes_found = False
                break
            etabs_node_names_for_plate_list.append(node_sm_id_to_etabs_name[sm_node_id])

        if not all_nodes_found:
            continue

        # Determine shell section name based on thickness (already converted to mm for definition)
        thickness_m = plate_obj.thickness
        thickness_mm_val = thickness_m * M
        shell_sec_name_for_plate = f"Shell_T{thickness_mm_val:.2f}".replace(".", "_")

        if shell_sec_name_for_plate not in defined_shell_sections:
            print(
                f"Warning: Skipping plate {sm_plate_name_req}. Shell section '{shell_sec_name_for_plate}' was not defined or failed definition."
            )
            continue

        num_points_in_plate = len(etabs_node_names_for_plate_list)
        temp_name = "temp_name"

        # AddByPoint for areas returns a tuple: (ActualNameAssignedByETABS, ReturnCode)
        # Note: ETABS API requires node names as a list/tuple of strings.
        nodes, etabs_assigned_plate_name, ret_code = SapModel.AreaObj.AddByPoint(
            num_points_in_plate,
            etabs_node_names_for_plate_list,
            temp_name,
            shell_sec_name_for_plate,
            UserName=sm_plate_name_req,
        )
        op_result(
            ret_code,
            f"Add Plate (req: {sm_plate_name_req}, ETABS: {etabs_assigned_plate_name}) with nodes {etabs_node_names_for_plate_list} section '{shell_sec_name_for_plate}'",
        )

        # 为成功创建的板添加均布荷载 -30 kN/m² (沿局部3轴方向)
        if ret_code == 0:
            # 获取ETABS中实际使用的名称
            plate_name_to_use = etabs_assigned_plate_name

            # 荷载参数设置
            load_pattern_name = "Pressure"  # 荷载工况名
            load_direction = 3  # 3表示局部3轴方向（通常是板的法线方向）
            load_value = -30.0e3 * N / (M * M)  # 荷载大小，kN/m²，负值表示向下压力
            if plate_obj.on_face == "top":
                load_value = -load_value

            # 应用均布荷载
            load_ret = SapModel.AreaObj.SetLoadUniform(
                plate_name_to_use,  # 板名称
                load_pattern_name,  # 荷载工况
                load_value,  # 荷载值
                load_direction,  # 荷载方向
                Replace=True,  # 替换已有荷载
                CSys="Local",  # 空字符串表示使用局部坐标系
            )
            op_result(load_ret, f"Add uniform load {load_value} kN/m² to plate {plate_name_to_use}")

    # 8. set mass source to only additional mass

    # Refresh View
    SapModel.View.RefreshView(Window=0, Zoom=False)  # 0 for all windows, False for not regenerating zoom

    print("\nETABS model creation process finished.")
    return SapModel, myETABSObject, created_new_instance


# Example usage:
if __name__ == "__main__":
    print("--- Starting ETABS Model Generation Example ---")

    from src.utils.model_utils import create_structural_model, load_config

    config = load_config("config.yaml")
    structural_model_instance = create_structural_model(config)

    if structural_model_instance:
        # Attempt to build the ETABS model
        # Set attach_mode=True if ETABS is already open with a model you want to modify (less common for full build)
        # Set program_path if ETABS.exe is not in the standard system path recognized by comtypes
        SapModel_obj, ETABS_app_obj, was_created_new = build_etabs_model(
            structural_model_instance,
            attach_mode=True,
            program_path="",  # e.g., "C:\Program Files\Computers and Structures\ETABS 20\ETABS.exe"
            program_target="ETABS",
        )

        if SapModel_obj:
            print("\nETABS model generation attempt completed.")
            # Save the model (optional)
            # Ensure the directory exists
            model_base_dir = os.path.join(os.getcwd(), "results", "etabs_models_output")
            if not os.path.exists(model_base_dir):
                try:
                    os.makedirs(model_base_dir)
                except OSError as e:
                    print(f"Error creating directory {model_base_dir}: {e}")
                    model_base_dir = os.getcwd()  # fallback to current dir

            model_filename = "GeneratedEtabsModel.EDB"
            model_full_path = os.path.join(model_base_dir, model_filename)

            print(f"Attempting to save ETABS model to: {model_full_path}")
            save_ret_code = SapModel_obj.File.Save(model_full_path)
            op_result(save_ret_code, f"Save model to {model_full_path}")

            # If you created a new instance of ETABS, you might want to close it.
            # Be careful with auto-closing if you attached or if user wants to inspect.
            if was_created_new and ETABS_app_obj:
                user_closes_etabs = True  # Set to False to auto-close
                if not user_closes_etabs:
                    print("Closing ETABS application...")
                    try:
                        ETABS_app_obj.ApplicationExit(
                            SaveChanges=False
                        )  # False = do not save changes on exit (already saved)
                    except Exception as e_exit_main:
                        print(f"Error trying to exit ETABS: {e_exit_main}")
                else:
                    print("ETABS application was started by this script. Please close it manually when done.")
            elif ETABS_app_obj:
                print("Attached to an existing ETABS instance. Please manage ETABS manually.")

        else:
            print("Failed to build or connect to ETABS model.")
    else:
        print("StructuralModel instance was not available. Skipping ETABS model generation.")

    print("--- ETABS Model Generation Example Finished ---")
