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


def circle_radii(radius: float, k: int):
    """Constant radius for a circle approximation."""
    return np.full(k, float(radius))


def ellipse_radii(a: float, b: float, k: int):
    """Radii for an ellipse centered at origin with semi-axes a (x) and b (y).

    r(theta) = 1 / sqrt((cos(theta)/a)^2 + (sin(theta)/b)^2)
    """
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    cos = np.cos(angles)
    sin = np.sin(angles)
    denom = (cos / a) ** 2 + (sin / b) ** 2
    radii = 1.0 / np.sqrt(denom)
    return radii


def regular_polygon_radii(n_sides: int, circumradius: float, k: int, rotation: float = 0.0):
    """Compute radial distances from center to edges of a regular n-gon.

    - n_sides: number of polygon sides (3 = triangle)
    - circumradius: distance from center to vertices
    - k: number of sample angles
    - rotation: rotation in degrees applied to the polygon

    Uses apothem-based formula: r(theta) = apothem / cos(alpha)
    where alpha is the signed angle between ray and nearest edge normal.
    """
    if n_sides < 3:
        raise ValueError("n_sides must be >= 3")
    apothem = circumradius * np.cos(np.pi / n_sides)
    sector = 2.0 * np.pi / n_sides
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    # apply rotation
    angles = angles - np.deg2rad(rotation)
    # compute alpha = angle relative to sector center in [-sector/2, sector/2]
    mod_angles = (angles + sector / 2.0) % sector - sector / 2.0
    # avoid numerical issues when cos(alpha) is very small
    cos_alpha = np.cos(mod_angles)
    cos_alpha = np.where(np.abs(cos_alpha) < 1e-8, 1e-8, cos_alpha)
    radii = apothem / cos_alpha
    return radii


def star_radii(points: int, inner_r: float, outer_r: float, k: int):
    """Approximate a star-shaped polygon (radial sinusoidal modulation).

    - points: number of star points
    - inner_r, outer_r: inner and outer radii
    - k: samples
    """
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    amp = (outer_r - inner_r) / 2.0
    mid = (outer_r + inner_r) / 2.0
    radii = mid + amp * np.cos(points * angles)
    # ensure non-negative
    radii = np.maximum(radii, 1e-6)
    return radii


if __name__ == "__main__":
    # Simple multi-sample visualization demo: draw several random candidates
    import os
    import matplotlib.pyplot as plt

    """Draw one example per generator function with representative parameters.

    Shows: rectangle, circle, ellipse, regular polygon, star, and a random sample.
    """
    corridor = construct_corridor()
    examples = [
        ("rectangle_radii", rectangle_radii, {"width": 1.3, "height": 0.8, "k": 24}),
        ("circle_radii", circle_radii, {"radius": 0.9, "k": 24}),
        ("ellipse_radii", ellipse_radii, {"a": 1.1, "b": 0.6, "k": 24}),
        ("regular_polygon_radii", regular_polygon_radii, {"n_sides": 5, "circumradius": 1.0, "k": 24, "rotation": 0.0}),
        ("star_radii", star_radii, {"points": 5, "inner_r": 0.4, "outer_r": 1.0, "k": 24}),
        ("random_radii", generate_random_radii, {"k": 24, "r_min": 0.2, "r_max": 1.4, "seed": 123}),
    ]

    import math
    import matplotlib.pyplot as plt

    n = len(examples)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    else:
        axs = axs.flatten()

    for i, (name, func, kwargs) in enumerate(examples):
        ax = axs[i]
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_frame_on(False)

        # draw corridor
        cx, cy = corridor.exterior.xy
        ax.fill(cx, cy, color="lightgray", alpha=0.6)

        # build radii depending on signature
        try:
            if name == "rectangle_radii":
                radii = func(kwargs["width"], kwargs["height"], kwargs["k"])
            elif name == "circle_radii":
                radii = func(kwargs["radius"], kwargs["k"])
            elif name == "ellipse_radii":
                radii = func(kwargs["a"], kwargs["b"], kwargs["k"])
            elif name == "regular_polygon_radii":
                radii = func(kwargs["n_sides"], kwargs["circumradius"], kwargs["k"], kwargs.get("rotation", 0.0))
            elif name == "star_radii":
                radii = func(kwargs["points"], kwargs["inner_r"], kwargs["outer_r"], kwargs["k"])
            else:
                # random or other
                radii = func(**kwargs)
        except Exception as exc:
            ax.text(0.5, 0.5, f"error: {exc}", ha="center", va="center", transform=ax.transAxes)
            continue

        # optional smoothing for visualization clarity
        radii_s = smooth_radii(radii, window=1)
        poly = radii_to_polygon(radii_s)
        valid = validate_polygon(poly)

        if not valid:
            try:
                poly = poly.buffer(0)
                valid = validate_polygon(poly)
            except Exception:
                valid = False

        if valid:
            placed = place_polygon_at_start(poly, corridor, x_offset=0.0)
            px, py = placed.exterior.xy
            ax.fill(px, py, color="teal", alpha=0.8)
        else:
            ax.text(0.5, 0.5, "invalid", ha="center", va="center", transform=ax.transAxes)

        title_params = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "k")
        ax.set_title(f"{name} ({title_params})", fontsize=9)

    # hide any remaining axes
    for j in range(len(examples), len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()