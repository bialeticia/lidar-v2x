from __future__ import annotations
import os
import numpy as np


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _to_xyz(points: np.ndarray) -> np.ndarray:
    if points is None:
        raise ValueError("points=None (nenhuma nuvem disponível)")
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points precisa ser Nx3 ou Nx4; veio {pts.shape}")
    return pts[:, :3].astype(np.float64, copy=False)


def save_pointcloud_png(points: np.ndarray, out_png: str, *,
                        width: int = 1280, height: int = 720) -> None:
    import open3d as o3d

    xyz = _to_xyz(points)
    _ensure_dir(out_png)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(out_png, do_render=True)
    vis.destroy_window()
