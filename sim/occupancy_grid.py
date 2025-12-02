from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

try:
    import cv2
except Exception:
    cv2 = None

def _fit_plane(xyz: np.ndarray) -> tuple[np.ndarray, float]:

    c = xyz.mean(0)
    _, _, vh = np.linalg.svd(xyz - c, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-9)
    d = -float(n @ c)
    return n, d

def remove_ground(points_xyz: np.ndarray,
                  max_dist: float = 0.18,
                  near_r: float = 22.0,
                  max_tilt_deg: float = 22.0,
                  enabled: bool = True) -> np.ndarray:

    if not enabled or points_xyz.size == 0:
        return np.ones((points_xyz.shape[0],), dtype=bool)

    xyz = points_xyz[:, :3]
    r = np.linalg.norm(xyz[:, :2], 2, 1)
    base = xyz[r <= near_r]
    if base.shape[0] < 200:
        return np.ones((xyz.shape[0],), dtype=bool)

    n, d = _fit_plane(base)
    tilt = math.degrees(math.acos(np.clip(abs(n[2]), 0.0, 1.0)))
    if tilt > max_tilt_deg:
        return np.ones((xyz.shape[0],), dtype=bool)

    dist = np.abs(xyz @ n + d)
    return dist > max_dist

def _polar_nearest(points_xy: np.ndarray,
                   size_m: float,
                   deg_step: float = 0.5,
                   r_step: float = 0.15) -> np.ndarray:

    if points_xy.size == 0:
        return np.empty((0, 2))

    x, y = points_xy[:, 0], points_xy[:, 1]
    az = np.degrees(np.arctan2(y, x))
    r = np.hypot(x, y)

    az_bins = np.arange(-180.0, 180.0 + deg_step, deg_step)
    idx = np.clip(np.digitize(az, az_bins) - 1, 0, len(az_bins) - 2)

    num = len(az_bins) - 1
    min_r = np.full(num, np.inf, np.float32)
    seen = np.zeros(num, bool)

    for i in range(num):
        ri = r[idx == i]
        if ri.size:
            m = ri.min()
            if 0.0 < m < (size_m / 2 - 0.1):
                min_r[i] = m
                seen[i] = True

    if not seen.any():
        return np.empty((0, 2))

    rng_bins = np.arange(0.0, size_m / 2 + r_step, r_step)
    rq = np.clip(np.digitize(min_r[seen], rng_bins) - 1, 0, len(rng_bins) - 2)
    r_cent = (rng_bins[:-1] + rng_bins[1:]) * 0.5
    r_sel = r_cent[rq]

    az_cent = (az_bins[:-1] + az_bins[1:]) * 0.5
    th = np.radians(az_cent[seen])
    return np.stack([r_sel * np.cos(th), r_sel * np.sin(th)], 1)


# ----------------- grade ----------------- #

def build_grid(points: np.ndarray,
               *,
               rmin: float = 1.6,
               size_m: float = 50.0,
               res: float = 0.15,
               z_min: float = -3.0,
               z_max: float = 2.0,
               ego_mask_r: float = 3.2,
               ground_on: bool = True) -> tuple[np.ndarray, np.ndarray]:

    n = int(round(size_m / res))
    grid = np.zeros((n, n), np.uint8)

    if points is None or len(points) == 0:
        return grid, np.empty((0, 2))

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return grid, np.empty((0, 2))

    pts = pts[np.isfinite(pts).all(1)]
    if pts.size == 0:
        return grid, np.empty((0, 2))

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.hypot(x, y)

    keep = (
        (r >= rmin)
        & (np.abs(x) <= size_m / 2)
        & (np.abs(y) <= size_m / 2)
        & (z >= z_min)
        & (z <= z_max)
        & (r >= ego_mask_r)
    )
    pts = pts[keep]
    if pts.size == 0:
        return grid, np.empty((0, 2))

    keep_ng = remove_ground(pts, enabled=ground_on)
    pts = pts[keep_ng]
    if pts.size == 0:
        return grid, np.empty((0, 2))

    pts_xy = _polar_nearest(pts[:, :2], size_m=size_m, deg_step=0.5, r_step=0.15)
    if pts_xy.size == 0:
        return grid, np.empty((0, 2))

    half = size_m / 2
    ix = np.floor((pts_xy[:, 0] + half) / res).astype(int)
    iy = np.floor((pts_xy[:, 1] + half) / res).astype(int)
    valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    grid[iy[valid], ix[valid]] = 1
    return grid, pts_xy

# ----------------- modelo b  ----------------- #

def postprocess_binary(binary: np.ndarray,
                       *,
                       kernel: int = 5,
                       dilate_iter: int = 2,
                       close_iter: int = 1) -> np.ndarray:

    img = (binary.astype(np.uint8) * 255)
    if cv2 is None:
        return (img > 0).astype(np.uint8)

    k = np.ones((kernel, kernel), np.uint8)
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=close_iter)
    out = cv2.dilate(out, k, iterations=dilate_iter)
    _, thr = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thr > 0).astype(np.uint8)


# ----------------- entre modelos ----------------- #

def compute_metrics(bin_raw: np.ndarray,
                    bin_post: np.ndarray,
                    n_pts: int,
                    frame_idx: int):

    from .logger_csv import FrameMetrics

    a = (bin_raw > 0).astype(np.uint8)
    b = (bin_post > 0).astype(np.uint8)

    n_occ_raw = int(a.sum())
    n_occ_post = int(b.sum())

    inter = int((a & b).sum())
    union = int((a | b).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0

    density_ratio = float(n_occ_post) / float(max(n_occ_raw, 1))

    return FrameMetrics(
        frame=frame_idx,
        n_pts=n_pts,
        n_occ_raw=n_occ_raw,
        n_occ_post=n_occ_post,
        iou_raw_post=iou,
        density_ratio=density_ratio,
    )

def _save_scatter(points_xy: np.ndarray, size_m: float, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6), dpi=200)
    if points_xy.size:
        plt.scatter(points_xy[:, 0], points_xy[:, 1], s=6, alpha=0.9)
    half = size_m / 2
    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"* scatter salvo: {path}")

def save_grid_png(grid: np.ndarray, size_m: float, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    half = size_m / 2
    plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(
        grid,
        extent=[-half, half, -half, half],
        origin="lower",
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"* grade salva: {path}")