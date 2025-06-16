import numpy as np


def calc_basic_stresses(basic_forces: np.ndarray, t: float) -> np.ndarray:
    """
    basic_forces: force or moment along local x, y, z axis
    shell element stresses response: p11, p22, p12, m11, m22, m12, q1, q2
    t: plate thickness

    """
    p11, p22, p12, m11, m22, m12, q1, q2 = basic_forces

    sigma_x_plus = p11 / t + 6 * m11 / t**2
    sigma_x_minus = p11 / t - 6 * m11 / t**2
    sigma_y_plus = p22 / t + 6 * m22 / t**2
    sigma_y_minus = p22 / t - 6 * m22 / t**2
    tau_xy_plus = p12 / t + 6 * m12 / t**2
    tau_xy_minus = p12 / t - 6 * m12 / t**2
    tau_xz = 3 * q1 / (2 * t)
    tau_yz = 3 * q2 / (2 * t)
    return np.array(
        [sigma_x_plus, sigma_x_minus, sigma_y_plus, sigma_y_minus, tau_xy_plus, tau_xy_minus, tau_xz, tau_yz]
    )


def fmt_display_components(
    tabular_values: np.ndarray, components: list[str] = None, title: str = None
) -> np.ndarray:
    if title:
        print(title)
    # Print header
    print(" ".join(f"{comp:<12}" for comp in components))
    print("-" * (12 * len(components)))
    # Print values
    if tabular_values.ndim == 1:
        print(" ".join(f"{val:<12.4f}" for val in tabular_values))
    else:
        for row in tabular_values:
            print(" ".join(f"{val:<12.4f}" for val in row))
    print()
