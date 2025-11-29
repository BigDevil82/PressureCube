###################### this file defines units used across the program ###################
# when define a parameter with unit, append unit symbol to the parameter name
# so you will not care about the conversion between units
# e.g. when define elastic modulus of concrete, set E = 30 * GPa


#################################### some basic units ####################################
# ********(remember all quantities are defined with the following basic units)************
M = 1000  # length unit, meter
s = 1  # time unit, second
# kg = 1  # mass unit, kilogram
N = 1  # force unit

#################################### derived units ####################################
# with this conversion, you can easily denote a quantity with your familiar unit

mm = 1e-3 * M  # length unit

# N = kg * M / s**2  # force unit
kg = N * s**2 / M  # mass unit
kN = 1e3 * N  # force unit
Pa = N / M**2  # pressure unit
MPa = 1e6 * Pa  # pressure unit
GPa = 1e3 * MPa  # pressure unit

ton = 1e3 * kg  # mass unit


# unit conversion
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


# gravity
G_MM_S2 = 9806.65  # Gravitational acceleration in mm/s^2
G_M_S2 = 9.80665
