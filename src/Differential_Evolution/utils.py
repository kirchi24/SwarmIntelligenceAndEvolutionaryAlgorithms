import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate


"""
Some code to tackle the moving sofa problem:

- a function to construct an L-shaped corridor
- a function to construct the Hammersley sofa shape (a known solution to the problem)
- a function to move and rotate a polygon shape smoothly within the corridor
- a function to check feasibility of moving and rotating a shape within the corridor
- a function to animate the movement of the shape within the corridor

========================================================================================

**Important**: You're certainly not gonna end up with the Hammersley sofa shape in this
assignment, and that's perfectly fine. Don't get discouraged if your shape looks very
unexciting and very different. 


The key takeaways here are:
---------------------------
- initialization is NOT always trivial

- repeated runs of optimization algorithms are CRUCIAL to validate your results 
  (here, you know the Hammersley sofa is a solution, so you could benchmark your own
  findings against its area. In real-world problems, you often don't have a known
  solution, so repeated runs with different initializations are the only way to gain
  confidence in your results)

- you can penalize **whatever you want** in optimization problems if you have some 
  domain knowledge supporting it
  (here, you could (should) penalize out-of-bound shapes, shapes that don't rotate
  properly around the corner, shapes that have too much of a circular geometry, etc.)


"""


def construct_corridor(
    corridor_width: float = 1.5,
    horizontal_length: float = 6.01,
    vertical_length: float = 6.01,
):
    # constructs an L-shaped corridor as a shapely polygon

    horizontal_leg = box(-0.01, -0.01, horizontal_length, corridor_width)
    vertical_leg = box(
        horizontal_length - corridor_width, -0.01, horizontal_length, vertical_length
    )
    full_corridor = horizontal_leg.union(vertical_leg)
    return full_corridor


def hammersley_sofa(disk_radius: float = 0.98, number_points: int = 30):
    # constructs the Hammersley sofa shape as a shapely polygon

    rectangle_width = 4 / np.pi
    removed_disk_radius = 2 / np.pi

    theta_left = np.linspace(np.pi / 2, np.pi, number_points)
    x_left = disk_radius * np.cos(theta_left)
    y_left = disk_radius * np.sin(theta_left)

    x_bottom_left = np.linspace(-disk_radius, 0, number_points)
    y_bottom_left = np.zeros(number_points)

    theta_bottom = np.linspace(np.pi, 0, number_points)
    x_bottom_bow = -removed_disk_radius + removed_disk_radius * np.cos(theta_bottom)
    y_bottom_bow = removed_disk_radius * np.sin(theta_bottom)

    x_bottom_right = np.linspace(
        2 * removed_disk_radius, disk_radius + 2 * removed_disk_radius, number_points
    )
    y_bottom_right = np.zeros(number_points)

    theta_right = np.linspace(0, np.pi / 2, number_points)
    x_offset = disk_radius + 2 * removed_disk_radius - 1
    x_right = x_offset + disk_radius * np.cos(theta_right)
    y_right = disk_radius * np.sin(theta_right)

    coords = []
    coords += list(zip(x_left, y_left))
    coords += list(zip(x_bottom_left, y_bottom_left))
    coords += list(zip(x_bottom_bow + rectangle_width, y_bottom_bow))
    coords += list(zip(x_bottom_right, y_bottom_right))
    coords += list(zip(x_right, y_right))

    # move into the corridor
    sofa_polygon = Polygon(coords)
    sofa_polygon = rotate(sofa_polygon, 180, use_radians=False)
    sofa_polygon = translate(sofa_polygon, xoff=+0.98, yoff=0)

    return sofa_polygon


def move_and_rotate_smooth(
    corridor: Polygon,
    polygon: Polygon,
    step_size: float = 0.05,
    rotation_increment: float = 1,
):
    # move and rotate your polygon smoothly within the corridor

    path = []
    current_poly = polygon
    total_rotation = 0
    finished_rotation = False

    if not corridor.covers(current_poly):
        return False, 0.0, path

    while True:

        if not finished_rotation:
            moved = translate(current_poly, xoff=step_size, yoff=0)

            if moved.within(corridor):
                current_poly = moved
                path.append(current_poly)

            else:
                pivot = (current_poly.bounds[2], current_poly.bounds[3])
                rotated = rotate(
                    current_poly, rotation_increment, origin=pivot, use_radians=False
                )
                total_rotation += rotation_increment

                _, miny, maxx, maxy = rotated.bounds
                x_shift = 0
                y_shift = 0

                if maxx > max(corridor.boundary.xy[0]):
                    x_shift = max(corridor.boundary.xy[0]) - maxx
                if miny < 0:
                    y_shift = -miny
                if maxy > max(corridor.boundary.xy[1]):
                    y_shift = max(corridor.boundary.xy[1]) - maxy

                rotated = translate(rotated, xoff=x_shift, yoff=y_shift)

                if not rotated.within(corridor):
                    maximum_rotation = total_rotation / 90
                    return False, maximum_rotation, path  # cannot rotate further, stop
                else:
                    current_poly = rotated
                    path.append(current_poly)
                    maximum_rotation = total_rotation / 90

                if total_rotation >= 90:
                    maximum_rotation = 1
                    finished_rotation = True

        else:
            moved = translate(current_poly, xoff=0, yoff=step_size)

            if moved.within(corridor):
                current_poly = moved
                path.append(current_poly)
            else:
                break

    return True, maximum_rotation, path


def check_feasibility(
    corridor: Polygon, polygon: Polygon, rotation_increment: float = 3
):
    # reuses the logic of move_and_rotate_smooth to just check feasibility

    _, _, maxx_c, _ = corridor.bounds
    _, _, maxx_p, _ = polygon.bounds
    shift_x = maxx_c - maxx_p
    current = translate(polygon, xoff=shift_x)

    if not current.within(corridor):
        return False, 0.0

    total_rotation = 0.0
    while total_rotation < 90:

        moved = translate(current, xoff=0.2, yoff=0)
        if moved.within(corridor):
            current = moved

        else:
            pivot = (moved.bounds[2], moved.bounds[3])
            rotated = rotate(moved, rotation_increment, origin=pivot, use_radians=False)
            total_rotation += rotation_increment

            _, miny, maxx, maxy = rotated.bounds
            x_shift = 0
            y_shift = 0

            if maxx > max(corridor.boundary.xy[0]):
                x_shift = max(corridor.boundary.xy[0]) - maxx
            if miny < 0:
                y_shift = -miny
            if maxy > max(corridor.boundary.xy[1]):
                y_shift = max(corridor.boundary.xy[1]) - maxy

            rotated = translate(rotated, xoff=x_shift, yoff=y_shift)

            if not rotated.within(corridor):
                maximum_rotation = total_rotation / 90
                return False, maximum_rotation
            else:
                current = rotated
                maximum_rotation = total_rotation / 90

            if total_rotation >= 90:
                maximum_rotation = 1

    return True, 1.0


def animate_shape(
    corridor: Polygon, shape: Polygon, path: list[Polygon], interval=1, repeat=False
):
    # animate the movement of the shape within the corridor
    fig, ax = plt.subplots()

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_frame_on(False)

    cx, cy = corridor.exterior.xy
    ax.set_xlim(min(cx) - 0.5, max(cx) + 1)
    ax.set_ylim(min(cy) - 0.5, max(cy) + 1)
    ax.fill(cx, cy, color="lightgray", alpha=0.5)
    sx, sy = shape.exterior.xy
    ax.fill(sx, sy, color="blue", alpha=0.2)

    shape_patch = ax.fill([], [], color="blue", alpha=0.4)[0]

    def animate(frame):
        poly = path[frame]
        x, y = poly.exterior.xy
        shape_patch.set_xy(np.column_stack([x, y]))
        return (shape_patch,)

    animation = FuncAnimation(
        fig, animate, frames=len(path), blit=True, interval=interval, repeat=repeat
    )

    plt.show()
    return animation

def smoothness_penalty(shape):
    x, y = shape.exterior.xy
    pts = np.column_stack([x, y])
    diffs = np.linalg.norm(np.roll(pts, 1, axis=0) - pts, axis=1)
    return np.std(diffs)  # kleine Schwankungen = gut

def bilateral_symmetry_penalty(shape):
    x, y = shape.exterior.xy
    pts = np.column_stack([x, y])
    n = len(pts)
    mids = n // 2
    diffs = np.linalg.norm(pts - np.roll(pts, mids, axis=0), axis=1)
    return np.mean(diffs)

from shapely.geometry import Polygon

def concavity_penalty(shape):
    hull = shape.convex_hull
    # Fläche, die dem Hull fehlt → Maß für Konkavität
    return hull.area - shape.area

def aspect_ratio_penalty(shape):
    minx, miny, maxx, maxy = shape.bounds
    w = maxx - minx
    h = maxy - miny
    ratio = max(w, h) / max(min(w, h), 1e-6)
    return max(0, ratio - 3)  # erlaubt Aspect Ratio bis 3



def objective_function(
    corridor: Polygon, 
    shape: Polygon, 
    weights = {
        "rotation": 5.0,       
        "placement": 1.0,      
        "area": 1.0,           
        "noncircular": 0.1,    
        "smoothness": 0.0,
        "symmetry": 0.0,
        "concavity": 0.0,
        "aspect": 0.0,
    }
):

    # ---------- area ----------
    area = shape.area

    # ---------- placement penalty ----------
    if corridor.covers(shape):
        placement_penalty = 0.0
    else:
        outside_area = shape.difference(corridor)
        placement_penalty = outside_area.area if not outside_area.is_empty else 0

    # ---------- rotation feasibility ----------
    feasible, max_rot_fraction = check_feasibility(corridor, shape)
    rotation_penalty = 0.0 if feasible else (1 - max_rot_fraction)

    # ---------- noncircularity ----------
    cx, cy = shape.centroid.x, shape.centroid.y
    x, y = shape.exterior.xy
    radial = np.sqrt((np.array(x) - cx)**2 + (np.array(y) - cy)**2)
    noncircularity = np.std(radial)

    # ---------- smoothness penalty ----------
    smooth_pen = smoothness_penalty(shape)

    # ---------- anti-bow-tie symmetry penalty ----------
    symmetry_pen = bilateral_symmetry_penalty(shape)

    # ---------- concavity penalty ----------
    concave_pen = concavity_penalty(shape)

    # ---------- aspect ratio penalty ----------
    aspect_pen = aspect_ratio_penalty(shape)

    # combine everything
    total_cost = (
        weights["rotation"] * rotation_penalty +
        weights["placement"] * placement_penalty +
        weights["smoothness"] * smooth_pen +
        weights["symmetry"] * symmetry_pen +
        weights["concavity"] * concave_pen +
        weights["aspect"] * aspect_pen -
        weights["area"] * area -
        weights["noncircular"] * noncircularity
    )

    return total_cost



if __name__ == "__main__":
    # test the hammersley sofa in the corridor
    corridor = construct_corridor()
    sofa_shape = hammersley_sofa()
    possible, max_rot, path = move_and_rotate_smooth(corridor, sofa_shape)
    ani = animate_shape(corridor, sofa_shape, path, interval=5)
