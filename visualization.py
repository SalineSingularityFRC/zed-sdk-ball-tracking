"""3D visualization of the AprilTag localizer using VTK.

Runs the CameraLocalizer against a live ZED camera and renders:
  - the ground-truth AprilTag poses from tags.json
  - the estimated camera pose, updated each frame
"""

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import vtk

from localization.camera_localizer import CameraLocalizer, AprilTagOnlyLocalizer
from localization.ekf import load_tag_world_poses


def pose_to_vtk_matrix(pose) -> vtk.vtkMatrix4x4:
    m = np.asarray(pose.matrix(), dtype=float)
    vm = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vm.SetElement(i, j, float(m[i, j]))
    return vm


def make_axes_actor(length: float) -> vtk.vtkAxesActor:
    a = vtk.vtkAxesActor()
    a.SetTotalLength(length, length, length)
    a.SetShaftTypeToCylinder()
    a.SetCylinderRadius(0.03)
    a.AxisLabelsOff()
    return a


def make_tag_actor(pose, tag_id: int, size: float) -> vtk.vtkAssembly:
    """Flat square in the tag's local XY plane, plus a small axes triad."""
    assembly = vtk.vtkAssembly()

    plane = vtk.vtkPlaneSource()
    half = size / 2.0
    plane.SetOrigin(-half, -half, 0.0)
    plane.SetPoint1( half, -half, 0.0)
    plane.SetPoint2(-half,  half, 0.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())
    square = vtk.vtkActor()
    square.SetMapper(mapper)
    square.GetProperty().SetColor(0.95, 0.95, 0.95)
    square.GetProperty().SetEdgeVisibility(True)
    square.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
    square.GetProperty().SetLineWidth(2)
    assembly.AddPart(square)

    assembly.AddPart(make_axes_actor(size))
    assembly.SetUserMatrix(pose_to_vtk_matrix(pose))
    return assembly


def make_tag_label(pose, tag_id: int, size: float) -> vtk.vtkBillboardTextActor3D:
    label = vtk.vtkBillboardTextActor3D()
    label.SetInput(f"id{tag_id}")
    t = pose.translation()
    label.SetPosition(float(t[0]), float(t[1]), float(t[2]) + size)
    tp = label.GetTextProperty()
    tp.SetFontSize(14)
    tp.SetColor(1.0, 1.0, 0.2)
    tp.BoldOn()
    return label


def make_camera_actor() -> vtk.vtkAssembly:
    assembly = vtk.vtkAssembly()

    cone = vtk.vtkConeSource()
    cone.SetHeight(0.25)
    cone.SetRadius(0.12)
    cone.SetResolution(4)
    cone.SetDirection(0.0, 0.0, 1.0)
    cone.SetCenter(0.0, 0.0, 0.125)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cone.GetOutputPort())
    cone_actor = vtk.vtkActor()
    cone_actor.SetMapper(mapper)
    cone_actor.GetProperty().SetColor(0.2, 0.8, 1.0)
    cone_actor.GetProperty().SetOpacity(0.55)
    assembly.AddPart(cone_actor)

    assembly.AddPart(make_axes_actor(0.25))
    return assembly



TARGET_TAG_IDS = (2, 4, 5, 10)
TARGET_RADIUS_M = 0.25
TARGET_PLANE_OFFSET_M = 0.2


def compute_target_center(tag_world_poses: dict,
                          tag_ids=TARGET_TAG_IDS) -> tuple[float, float, float] | None:
    """Mean (x, y, z) of the target AprilTags in the world frame."""
    positions = []
    for tid in tag_ids:
        pose = tag_world_poses.get(tid)
        if pose is None:
            continue
        t = pose.translation()
        positions.append((float(t[0]), float(t[1]), float(t[2])))
    if not positions:
        return None
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    cz = sum(p[2] for p in positions) / len(positions)
    return cx, cy, cz


def target_plane_z(tag_world_poses: dict) -> float | None:
    center = compute_target_center(tag_world_poses)
    if center is None:
        return None
    return center[2] + TARGET_PLANE_OFFSET_M


def make_center_circle_actor(tag_world_poses: dict,
                             radius: float = TARGET_RADIUS_M) -> vtk.vtkActor | None:
    """Circle parallel to the ground, at TARGET_PLANE_OFFSET_M above the mean
    tag z, centered on the mean XY of the target tags."""
    center = compute_target_center(tag_world_poses)
    if center is None:
        return None
    cx, cy, cz = center
    plane_z = cz + TARGET_PLANE_OFFSET_M

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    n = 128
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        pts.InsertNextPoint(cx + radius * np.cos(theta),
                            cy + radius * np.sin(theta),
                            plane_z)
    lines.InsertNextCell(n + 1)
    for i in range(n):
        lines.InsertCellPoint(i)
    lines.InsertCellPoint(0)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.85, 0.2)
    actor.GetProperty().SetLineWidth(3)
    return actor


def make_ground_grid(half_extent: float, step: float) -> vtk.vtkActor:
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    idx = 0
    n = int(half_extent / step)
    for i in range(-n, n + 1):
        k = i * step
        pts.InsertNextPoint(k, -half_extent, 0.0)
        pts.InsertNextPoint(k,  half_extent, 0.0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx); lines.InsertCellPoint(idx + 1)
        idx += 2

        pts.InsertNextPoint(-half_extent, k, 0.0)
        pts.InsertNextPoint( half_extent, k, 0.0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx); lines.InsertCellPoint(idx + 1)
        idx += 2

    grid = vtk.vtkPolyData()
    grid.SetPoints(pts)
    grid.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(grid)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.25, 0.25, 0.32)
    return actor


def compute_track_miss(pts, center_xy, plane_z):
    """Return the (x, y) where the track crosses z=plane_z going downward, and
    the horizontal distance from ``center_xy``. Returns None if no downward
    crossing exists."""
    cx, cy = center_xy
    for i in range(len(pts) - 1):
        z0 = pts[i][2]
        z1 = pts[i + 1][2]
        if z0 >= plane_z > z1:
            s = (z0 - plane_z) / (z0 - z1) if z0 != z1 else 0.0
            x = pts[i][0] + s * (pts[i + 1][0] - pts[i][0])
            y = pts[i][1] + s * (pts[i + 1][1] - pts[i][1])
            miss = float(np.hypot(x - cx, y - cy))
            return (float(x), float(y), miss)
    return None


def show_miss_plot(misses: list) -> None:
    """2D plot of target-plane miss distances. ``misses`` is a list of dicts
    with keys ``x``, ``y``, ``miss``, ``color``, ``index``."""
    if not misses:
        print("No downward crossings of the target plane — nothing to plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping miss plot.")
        return

    cx, cy = misses[0]['center']
    plane_z = misses[0]['plane_z']

    fig, (ax_xy, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax_xy.plot(cx + TARGET_RADIUS_M * np.cos(theta),
               cy + TARGET_RADIUS_M * np.sin(theta),
               color=(1.0, 0.7, 0.1), lw=2, label=f"target r={TARGET_RADIUS_M:.2f} m")
    ax_xy.plot(cx, cy, marker='+', color=(1.0, 0.7, 0.1), markersize=12)
    for m in misses:
        ax_xy.plot(m['x'], m['y'], marker='o', color=m['color'], markersize=10,
                   markeredgecolor='black')
        ax_xy.annotate(f"#{m['index']}", (m['x'], m['y']),
                       textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title(f'Crossings at z = {plane_z:.2f} m (descending)')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc='best', fontsize=9)

    indices = [m['index'] for m in misses]
    values = [m['miss'] for m in misses]
    colors = [m['color'] for m in misses]
    bars = ax_bar.bar(range(len(misses)), values, color=colors, edgecolor='black')
    ax_bar.axhline(TARGET_RADIUS_M, color=(1.0, 0.7, 0.1), ls='--',
                   label=f"target radius ({TARGET_RADIUS_M:.2f} m)")
    ax_bar.set_xticks(range(len(misses)))
    ax_bar.set_xticklabels([f"#{i}" for i in indices])
    ax_bar.set_ylabel('Miss distance from center (m)')
    ax_bar.set_title('Per-track miss amount')
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.legend(loc='best', fontsize=9)
    for bar, v in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    plt.show()


def show_final_tracks(tag_world_poses: dict, tag_size: float, tracks_world: list) -> None:
    """Open a blocking VTK window showing AprilTags + static ball tracks in world frame.

    ``tracks_world`` is a list of dicts, each with keys:
      - ``points``: iterable of ``(x, y, z)`` world-frame tuples (meters)
      - ``color``: ``(r, g, b)`` tuple in 0..1
    """
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.08, 0.08, 0.12)
    renderer.AddActor(make_ground_grid(half_extent=9.0, step=1.0))

    world_axes = vtk.vtkAxesActor()
    world_axes.SetTotalLength(1.0, 1.0, 1.0)
    renderer.AddActor(world_axes)

    for tag_id, pose in tag_world_poses.items():
        renderer.AddActor(make_tag_actor(pose, tag_id, tag_size))
        renderer.AddActor(make_tag_label(pose, tag_id, tag_size))

    circle = make_center_circle_actor(tag_world_poses)
    if circle is not None:
        renderer.AddActor(circle)

    target_center3 = compute_target_center(tag_world_poses)
    if target_center3 is not None:
        target_center_xy = (target_center3[0], target_center3[1])
        target_plane = target_center3[2] + TARGET_PLANE_OFFSET_M
    else:
        target_center_xy = None
        target_plane = None
    misses = []

    for idx, track in enumerate(tracks_world):
        pts = list(track.get('points', []))
        if len(pts) < 2:
            continue
        color = track.get('color', (1.0, 1.0, 1.0))

        if target_center_xy is not None:
            result = compute_track_miss(pts, target_center_xy, target_plane)
            if result is not None:
                mx, my, md = result
                misses.append({
                    'index': idx,
                    'x': mx,
                    'y': my,
                    'miss': md,
                    'color': color,
                    'center': target_center_xy,
                    'plane_z': target_plane,
                })

        render_pts = [p for p in pts if p[2] >= 0.0]
        if len(render_pts) < 2:
            continue

        vtk_points = vtk.vtkPoints()
        vtk_lines = vtk.vtkCellArray()
        for i, (x, y, z) in enumerate(render_pts):
            vtk_points.InsertNextPoint(float(x), float(y), float(z))
            if i > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i - 1)
                line.GetPointIds().SetId(1, i)
                vtk_lines.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(vtk_lines)

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(poly)
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(*color)
        line_actor.GetProperty().SetLineWidth(3)
        renderer.AddActor(line_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1280, 800)
    render_window.SetWindowName("Final Tracks (field frame)")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()

    show_miss_plot(misses)


class LocalizerVisualizer:
    def __init__(self, localizer: CameraLocalizer, tag_world_poses: dict, tag_size: float):
        self.localizer = localizer

        self._pose_lock = threading.Lock()
        self._latest_pose = None
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.12)

        self.renderer.AddActor(make_ground_grid(half_extent=9.0, step=1.0))

        world_axes = vtk.vtkAxesActor()
        world_axes.SetTotalLength(1.0, 1.0, 1.0)
        self.renderer.AddActor(world_axes)

        for tag_id, pose in tag_world_poses.items():
            self.renderer.AddActor(make_tag_actor(pose, tag_id, tag_size))
            self.renderer.AddActor(make_tag_label(pose, tag_id, tag_size))

        circle = make_center_circle_actor(tag_world_poses)
        if circle is not None:
            self.renderer.AddActor(circle)

        self.camera_actor = make_camera_actor()
        self.renderer.AddActor(self.camera_actor)

        self._path_points = vtk.vtkPoints()
        self._path_lines = vtk.vtkCellArray()
        self._path_poly = vtk.vtkPolyData()
        self._path_poly.SetPoints(self._path_points)
        self._path_poly.SetLines(self._path_lines)
        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(self._path_poly)
        path_actor = vtk.vtkActor()
        path_actor.SetMapper(path_mapper)
        path_actor.GetProperty().SetColor(1.0, 0.45, 0.45)
        path_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(path_actor)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1280, 800)
        self.render_window.SetWindowName("Localizer Visualization")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.renderer.ResetCamera()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            pose = self.localizer.step()
            if pose is None:
                continue
            with self._pose_lock:
                self._latest_pose = pose

    def _on_timer(self, _obj, _event):
        with self._pose_lock:
            pose = self._latest_pose
            self._latest_pose = None
        if pose is None:
            return

        self.camera_actor.SetUserMatrix(pose_to_vtk_matrix(pose))

        t = pose.translation()
        pid = self._path_points.InsertNextPoint(float(t[0]), float(t[1]), float(t[2]))
        if pid > 0:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid - 1)
            line.GetPointIds().SetId(1, pid)
            self._path_lines.InsertNextCell(line)
        self._path_points.Modified()
        self._path_poly.Modified()

        self.render_window.Render()

    def run(self):
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self._on_timer)
        self.interactor.CreateRepeatingTimer(33)
        self.render_window.Render()
        self._worker.start()
        try:
            self.interactor.Start()
        finally:
            self._stop_event.set()
            self._worker.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="VTK visualization for the AprilTag localizer")
    parser.add_argument("--tags", type=str, default="localization/tags.json",
                        help="Path to tags.json with ground-truth AprilTag poses")
    parser.add_argument("--tag-size", type=float, default=0.1651,
                        help="AprilTag edge length in meters")
    parser.add_argument("--tags-only", action="store_true",
                        help="Visualize raw AprilTag camera pose without the EKF / ZED fusion")
    args = parser.parse_args()

    tags_path = Path(args.tags)
    if not tags_path.exists():
        print(f"Error: tags file not found at {tags_path}")
        return 1

    tag_world_poses = load_tag_world_poses(tags_path)
    if not tag_world_poses:
        print(f"Warning: no tags loaded from {tags_path}")

    if args.tags_only:
        localizer = AprilTagOnlyLocalizer(tag_size=args.tag_size, tags_path=str(tags_path))
    else:
        localizer = CameraLocalizer(tag_size=args.tag_size, tags_path=str(tags_path))

    viz = LocalizerVisualizer(localizer, tag_world_poses, args.tag_size)
    viz.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
