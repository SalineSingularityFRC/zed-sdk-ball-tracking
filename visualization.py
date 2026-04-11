"""3D visualization of the AprilTag localizer using VTK.

Runs the CameraLocalizer against a live ZED camera and renders:
  - the ground-truth AprilTag poses from tags.json
  - the estimated camera pose, updated each frame
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk

from localization.camera_localizer import CameraLocalizer
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


def make_field_actor(mesh_path: Path) -> vtk.vtkActor:
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        reader = vtk.vtkOBJReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Unsupported field mesh format: {suffix}")
    reader.SetFileName(str(mesh_path))
    reader.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SetFeatureAngle(45.0)
    normals.SetSplitting(False)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.78, 0.80, 0.84)
    prop.SetOpacity(0.35)
    prop.SetEdgeVisibility(True)
    prop.SetEdgeColor(0.10, 0.10, 0.12)
    prop.SetLineWidth(1)
    prop.SetAmbient(0.25)
    prop.SetDiffuse(0.65)
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


class LocalizerVisualizer:
    def __init__(self, localizer: CameraLocalizer, tag_world_poses: dict, tag_size: float,
                 field_path: Path | None = None):
        self.localizer = localizer

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.12)

        self.renderer.AddActor(make_ground_grid(half_extent=9.0, step=1.0))

        if field_path is not None:
            if field_path.exists():
                self.renderer.AddActor(make_field_actor(field_path))
            else:
                print(f"Warning: field mesh not found at {field_path}, skipping")

        world_axes = vtk.vtkAxesActor()
        world_axes.SetTotalLength(1.0, 1.0, 1.0)
        self.renderer.AddActor(world_axes)

        for tag_id, pose in tag_world_poses.items():
            self.renderer.AddActor(make_tag_actor(pose, tag_id, tag_size))
            self.renderer.AddActor(make_tag_label(pose, tag_id, tag_size))

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

    def _on_timer(self, _obj, _event):
        pose = self.localizer.step()
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
        self.interactor.CreateRepeatingTimer(30)
        self.render_window.Render()
        self.interactor.Start()


def main():
    parser = argparse.ArgumentParser(description="VTK visualization for the AprilTag localizer")
    parser.add_argument("--tags", type=str, default="localization/tags.json",
                        help="Path to tags.json with ground-truth AprilTag poses")
    parser.add_argument("--tag-size", type=float, default=0.1651,
                        help="AprilTag edge length in meters")
    parser.add_argument("--field", type=str, default="playing_field.obj",
                        help="Path to the playing-field mesh (.obj, .stl, or .ply). "
                             "Pre-convert playing_field.step once and commit it.")
    args = parser.parse_args()

    tags_path = Path(args.tags)
    if not tags_path.exists():
        print(f"Error: tags file not found at {tags_path}")
        return 1

    tag_world_poses = load_tag_world_poses(tags_path)
    if not tag_world_poses:
        print(f"Warning: no tags loaded from {tags_path}")

    localizer = CameraLocalizer(tag_size=args.tag_size, tags_path=str(tags_path))

    viz = LocalizerVisualizer(localizer, tag_world_poses, args.tag_size,
                              field_path=Path(args.field))
    viz.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
