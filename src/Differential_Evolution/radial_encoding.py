import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

# use the utils for corridor, feasibility and visualization
from utils import (
    construct_corridor,
    check_feasibility,
    move_and_rotate_smooth,
    animate_shape,
    objective_function,
)


def generate_random_radii(k=24, r_min=0.1, r_max=1.5, seed=None):
    """Generate a random radii vector of length k bounded by [r_min, r_max]."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.uniform(r_min, r_max, size=k)
    return np.random.uniform(r_min, r_max, size=k)


def smooth_radii(radii, window=3):
    """Simple moving-average smoothing for radii (1D convolution).

    Keeps length unchanged and preserves positive radii.
    """
    if window <= 1:
        return radii.copy()
    kernel = np.ones(window) / window
    # pad to keep same length (reflect padding)
    pad = window // 2
    padded = np.pad(radii, pad_width=pad, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # ensure same length
    if len(smoothed) != len(radii):
        smoothed = smoothed[: len(radii)]
    # enforce non-negative
    smoothed = np.maximum(smoothed, 1e-6)
    return smoothed


def radii_to_polygon(radii, center=(0.0, 0.0), rotate_deg=0.0, translate_xy=(0.0, 0.0)):
    """Convert radial representation to a Shapely Polygon.

    - radii: iterable of length K, radii values for angles theta_k = 0..2pi
    - center: (x,y) center point around which radii are interpreted
    - rotate_deg: degrees to rotate the built polygon
    - translate_xy: (tx,ty) additional translation applied after rotation
    Returns: shapely.geometry.Polygon (may be invalid if self-intersecting)
    """
    K = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    cx, cy = center
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)]
    poly = Polygon(pts)
    if rotate_deg != 0.0:
        poly = rotate(poly, rotate_deg, origin=center, use_radians=False)
    if translate_xy != (0.0, 0.0):
        tx, ty = translate_xy
        poly = translate(poly, xoff=tx, yoff=ty)
    # try to repair simple self-intersections via buffer(0)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def validate_polygon(poly):
    """Return True if polygon is valid and has positive area."""
    if poly is None:
        return False
    if poly.is_empty:
        return False
    if not poly.is_valid:
        return False
    if poly.area <= 0:
        return False
    return True


def place_polygon_against_corridor(poly, corridor):
    """Translate polygon so its rightmost bound aligns with corridor's right bound.

    This matches the positioning used in `check_feasibility`.
    """
    _, _, maxx_c, _ = corridor.bounds
    _, _, maxx_p, _ = poly.bounds
    shift_x = maxx_c - maxx_p
    return translate(poly, xoff=shift_x, yoff=0.0)


def place_polygon_at_start(poly, corridor, x_offset: float = 0.0, y_offset: float = 0.0):
    """Place polygon at the beginning (left side) of the corridor.

    Aligns the polygon's leftmost x to the corridor's left bound plus optional
    `x_offset`, and aligns the polygon's bottom (min y) to the corridor's
    bottom plus optional `y_offset`.
    """
    minx_c, miny_c, _, _ = corridor.bounds
    minx_p, miny_p, _, _ = poly.bounds
    shift_x = (minx_c + x_offset) - minx_p
    shift_y = (miny_c + y_offset) - miny_p
    return translate(poly, xoff=shift_x, yoff=shift_y)


def rectangle_radii(width: float, height: float, k: int):
    """Compute radii for a rectangle centered at origin.

    For each angle theta, computes intersection distance from center to rectangle
    boundary along direction (cos(theta), sin(theta)).
    """
    a = width / 2.0
    b = height / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    radii = np.empty(k, dtype=float)
    for i, th in enumerate(angles):
        c = np.cos(th)
        s = np.sin(th)
        t_candidates = []
        if abs(c) > 1e-12:
            t_candidates.append(a / abs(c))
        if abs(s) > 1e-12:
            t_candidates.append(b / abs(s))
        radii[i] = min(t_candidates) if t_candidates else 0.0
    return radii


if __name__ == "__main__":
    # Simple static visualization demo (no animation, no feasibility calculations)
    corridor = construct_corridor()

    K = 24
    r_min, r_max = 0.2, 1.4

    # generate a single candidate and show it
    radii = generate_random_radii(K, r_min=r_min, r_max=r_max, seed=42)
    radii = smooth_radii(radii, window=5)
    poly = radii_to_polygon(radii)

    if not validate_polygon(poly):
        print("Generated polygon is invalid — increase smoothing or adjust bounds.")
    else:
        # place the candidate at the beginning (left) of the corridor
        placed = place_polygon_at_start(poly, corridor, x_offset=0.0)

        # static matplotlib figure: corridor + placed polygon
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_frame_on(False)

        cx, cy = corridor.exterior.xy
        ax.fill(cx, cy, color="lightgray", alpha=0.6)

        px, py = placed.exterior.xy
        ax.fill(px, py, color="blue", alpha=0.7)

        ax.set_title("Radial-encoded candidate placed in corridor")
        plt.show()
