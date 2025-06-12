import enum
import sys

import comtypes.client

from src.analysis.unit import *


class MatType(enum.Enum):
    Steel = 1
    Concrete = 2
    NoDesign = 3
    Aluminum = 4
    ColdFormed = 5
    Rebar = 6
    Tendon = 7
    Masonry = 8


class Units_sys(enum.Enum):
    lb_in_F = 1
    lb_ft_F = 2
    kip_in_F = 3
    kip_ft_F = 4
    kN_mm_C = 5
    kN_m_C = 6
    kgf_mm_C = 7
    kgf_m_C = 8
    N_mm_C = 9
    N_m_C = 10
    Ton_mm_C = 11
    Ton_m_C = 12
    kN_cm_C = 13
    kgf_cm_C = 14
    N_cm_C = 15
    Ton_cm_C = 16


class ShellType(enum.Enum):
    ShellThin = 1
    ShellThick = 2
    Membrane = 3
    PlateThin_DO_NOT_USE = 4
    PlateThick_DO_NOT_USE = 5
    Layered = 6


class SlabType(enum.Enum):
    Slab = 0
    Drop = 1
    Stiff_DO_NOT_USE = 2
    Ribbed = 3
    Waffle = 4
    Mat = 5
    Footing = 6


def op_result(ret_code, op_description, only_print_failed=True):
    """Prints the result of an operation."""
    if ret_code == 0:
        if not only_print_failed:
            print(f"{op_description} successfully.")
    else:
        print(f"{op_description} failed. Return code: {ret_code}")


def handle_csi_api_errors(func):
    """Decorator to handle errors from CSI API calls."""

    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            ret_value = func(*args, **kwargs)

            code_to_check = ret_value
            # If function returns a tuple, the last integer is usually the error code
            if isinstance(ret_value, tuple):
                # Look for the integer return code, often the last element
                found_code = False
                for item in reversed(ret_value):  # Check from the end
                    if isinstance(item, int):
                        code_to_check = item
                        found_code = True
                        break
                if (
                    not found_code and len(ret_value) > 0 and isinstance(ret_value[0], int)
                ):  # Check first if not found at end
                    code_to_check = ret_value[0]

            if code_to_check == 0:
                print(f"CSI API operation {func_name} executed successfully.")
            else:
                print(f"CSI API operation {func_name} failed with return code: {code_to_check}")
            return ret_value
        except Exception as e:
            print(f"Exception during CSI API operation {func_name}: {e}")
            # Return a consistent failure signature if possible, or a specific error code
            # For functions expected to return a tuple (value, ..., code), this might need adjustment
            return -99  # Generic exception error code

    return wrapper


def create_csi_object(program_target="ETABS", attach_mode=False, specific_program_path=""):
    """
    Creates or attaches to an instance of ETABS or SAP2000.

    Args:
        program_target (str): "ETABS" or "SAP2000".
        attach_mode (bool): If True, attach to a running instance.
        specific_program_path (str): Optional path to the program executable.

    Returns:
        tuple: (SapModel, csi_application_object, created_new_instance)
               Returns (None, None, False) on failure to initialize.
    """
    csi_application_object = None
    SapModel = None
    created_new_instance = False

    program_target_upper = program_target.upper()
    helper_prog_id = ""
    object_prog_id = ""
    program_name_for_messages = ""

    if program_target_upper == "ETABS":
        helper_prog_id = "ETABSv1.Helper"
        object_prog_id = "CSI.ETABS.API.ETABSObject"
        program_name_for_messages = "ETABS"
    elif program_target_upper == "SAP2000":
        helper_prog_id = "SAP2000v21.Helper"
        object_prog_id = "CSI.SAP2000.API.SapObject"
        program_name_for_messages = "SAP2000"
    else:
        print(f"Unsupported program_target: {program_target}. Must be 'ETABS' or 'SAP2000'.")
        return None, None, False

    def _create_new_instance_logic():
        nonlocal csi_application_object, SapModel
        print(f"Attempting to create a new instance of {program_name_for_messages}...")
        try:
            helper = comtypes.client.CreateObject(helper_prog_id)
            # Note: QueryInterface for specific helper (e.g., comtypes.gen.ETABSv1.cHelper)
            # may be needed if late binding causes issues, but requires type libraries to be generated.
            if program_target_upper == "SAP2000":
                helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
            else:
                helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)

            if specific_program_path:
                print(f"Using specific path: {specific_program_path}")
                csi_application_object = helper.CreateObject(specific_program_path)
            else:
                print(f"Using ProgID: {object_prog_id}")
                csi_application_object = helper.CreateObjectProgID(object_prog_id)

            csi_application_object.ApplicationStart()
            SapModel = csi_application_object.SapModel
            op_result(0, f"Successfully created and started new {program_name_for_messages} instance")
            return True
        except (OSError, comtypes.COMError) as e:
            print(
                f"Cannot start a new instance of {program_name_for_messages} (path: '{specific_program_path}', ProgID: '{object_prog_id}'). Error: {e}"
            )
            return False

    if not attach_mode:
        if _create_new_instance_logic():
            created_new_instance = True
        else:
            print(
                f"Failed to create new {program_name_for_messages} instance. Check installation and licensing."
            )
            return None, None, False
    else:  # attach_mode is True
        print(
            f"Attempting to attach to a running instance of {program_name_for_messages} (ProgID: {object_prog_id})..."
        )
        try:
            csi_application_object = comtypes.client.GetActiveObject(object_prog_id)
            SapModel = csi_application_object.SapModel
            print(f"Successfully attached to running {program_name_for_messages} instance.")
            created_new_instance = False
        except (OSError, comtypes.COMError) as e_attach:
            print(
                f"No running instance of {program_name_for_messages} found to attach (Error: {e_attach}). Trying to create a new one..."
            )
            if _create_new_instance_logic():
                created_new_instance = True
            else:
                print(
                    f"Failed to create a new instance of {program_name_for_messages} after failing to attach."
                )
                return None, None, False

    if SapModel and csi_application_object:
        op_result(0, f"{program_name_for_messages} API initialized")
    else:  # Should be caught by earlier returns, but as a safeguard
        op_result(-1, f"{program_name_for_messages} API initialization failed critically")
        return None, None, False

    return SapModel, csi_application_object, created_new_instance


################################# define materials #################################
@handle_csi_api_errors
def define_Conc_Mat(sap_model, material_name, E, miu, themal_exp, conc_level="C30"):
    ret_add = sap_model.PropMaterial.AddMaterial(
        material_name, MatType.Concrete.value, "China", "GB", f"GB50010 {conc_level}"
    )
    if ret_add != 0:
        return ret_add

    ret_iso = sap_model.PropMaterial.SetMPIsotropic(material_name, E, miu, themal_exp)
    if ret_iso != 0:
        return ret_iso

    density_val = 2550 * kg / (M**3)
    ret_wgt = sap_model.PropMaterial.SetWeightAndMass(material_name, 2, density_val)
    return ret_wgt


@handle_csi_api_errors
def define_Steel_Mat(
    sap_model, material_name, E, miu, themal_exp, fy=345, fu=470, fye=380, fue=520, steel_level="Q345"
):
    # get material first
    ret_get = sap_model.PropMaterial.GetMaterial(material_name, 1)
    if ret_get[-1] != 0:
        ret_add = sap_model.PropMaterial.AddMaterial(
            material_name, MatType.Steel.value, "China", "GB", steel_level
        )
        if ret_add != 0:
            # Non-zero might mean it exists. Some APIs are fine, others require Get/Set.
            # For now, we print a note and proceed. The decorator will report failure if subsequent calls fail.
            print(
                f"Note: AddMaterial for '{material_name}' returned {ret_add}. Proceeding to set properties."
            )
            # Do not return ret_add here, allow SetMPIsotropic etc. to be attempted.

    ret_iso = sap_model.PropMaterial.SetMPIsotropic(material_name, E, miu, themal_exp)
    if ret_iso != 0:
        return ret_iso

    # SetOSteel is typically for ETABS. SAP2000 might need different calls for stress-strain.
    # If this call fails in SAP2000, the decorator will report it.
    ret_strength = sap_model.PropMaterial.SetOSteel(material_name, fy, fu, fye, fue, 1, 0, 0.015, 0.11, 0.17)
    if ret_strength != 0:
        return ret_strength

    density_val = 7850.0 * kg / (M**3)
    ret_wgt = sap_model.PropMaterial.SetWeightAndMass(material_name, 2, density_val)
    return ret_wgt


################################# define sections ###################################
@handle_csi_api_errors
def define_wall_sec(sap_model, wall_sec_name, shell_type, mat_name, thickness):
    ret = sap_model.PropArea.SetWall(
        wall_sec_name, 1, shell_type.value, mat_name, thickness
    )  # Use .value for enum
    return ret


@handle_csi_api_errors
def define_slab_sec(
    sap_model, floor_sec_name, slab_type, shell_type, mat_name, thickness, program_target="ETABS"
):
    if program_target == "ETABS":
        ret = sap_model.PropArea.SetSlab(floor_sec_name, slab_type, shell_type, mat_name, thickness)
    else:
        ret = sap_model.PropArea.SetShell(floor_sec_name, shell_type, mat_name, 0.0, thickness, thickness)
    return ret


@handle_csi_api_errors
def define_beam_sec(sap_model, beam_sec_name, mat_name, h, b):  # Rectangular beam
    ret = sap_model.PropFrame.SetRectangle(beam_sec_name, mat_name, h, b)
    return ret
