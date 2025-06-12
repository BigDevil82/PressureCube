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
