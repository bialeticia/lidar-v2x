from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import carla

from sim.sensors_lidar import add_lidar
from sim.attack_spoofer import SpooferClient, V2XEndpoint, SpoofPatterns
from sim.occupancy_grid import (
    build_grid,
    postprocess_binary,
    save_grid_png,
    _save_scatter,
)
from sim.logger_csv import CSVLogger

from scripts.lidar_visor import save_pointcloud_png

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figs"
LOG_DIR = PROJECT_ROOT / "logs"
PCD_DIR = PROJECT_ROOT / "outputs" / "pointclouds"

FIG_RAW = FIG_DIR / "grade_raw.png"
FIG_ACCUM = FIG_DIR / "grade_accum.png"
FIG_POST = FIG_DIR / "grade_intermediate.png"
FIG_SCATTER = FIG_DIR / "grade_scatter.png"


def wait_world_ready(client: carla.Client,
                     tries: int = 40,
                     delay: float = 1.5) -> carla.World:
    last = None
    for i in range(tries):
        try:
            w = client.get_world()
            _ = w.get_map()
            _ = w.get_blueprint_library()
            return w
        except Exception as e:
            last = e
            print(f"[wait] {i+1}/{tries}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Mundo não ficou pronto: {last}")


def _metrics_iou_fp_fn(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """
    Métricas por frame entre duas grades binárias.
    - IoU
    - FP: b=1 e a=0
    - FN: b=0 e a=1
    """
    a1 = (a == 1)
    b1 = (b == 1)

    inter = int(np.logical_and(a1, b1).sum())
    union = int(np.logical_or(a1, b1).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0

    fp = int(np.logical_and(b1, ~a1).sum())
    fn = int(np.logical_and(~b1, a1).sum())
    return iou, fp, fn


def main(runtime_s: float = 35.0, send_spoof: bool = True) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PCD_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"metrics_{ts}.csv"
    logger = CSVLogger(log_path)
    print(f"Métricas em: {log_path}")

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = wait_world_ready(client)

    bp = world.get_blueprint_library()
    car_bp = bp.filter("vehicle.tesla.model3")[0]
    spawns = world.get_map().get_spawn_points()
    if len(spawns) < 2:
        raise RuntimeError("Poucos spawn.")

    ego = world.try_spawn_actor(car_bp, spawns[0])
    if ego is None:
        raise RuntimeError("Falha ao spawnar o ego.")
    other = world.try_spawn_actor(car_bp, spawns[1])

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(False)
    tm.global_percentage_speed_difference(30.0)
    ego.set_autopilot(True, tm.get_port())
    if other:
        other.set_autopilot(True, tm.get_port())

    lidar = add_lidar(
        world,
        ego,
        channels=32,
        rotation_hz=10,
        pps=600_000,
        rng=70.0,
        upper_fov=10.0,
        lower_fov=-30.0,
    )

    spoofer = SpooferClient(V2XEndpoint())

    t0 = time.time()
    last_save = 0.0

    res = 0.15
    size_m = 50.0
    rmin = getattr(lidar, "rmin", 1.6)

    accum: np.ndarray | None = None
    max_accum_frames = 10
    decay_per_second = 0.1
    fps_est = 20.0
    decay_per_frame = decay_per_second / fps_est
    threshold_count = 1

    attack_start = 8.0
    attack_end = 10.5

    saved_baseline = False
    saved_attack = False
    saved_after = False

    ref_grid: np.ndarray | None = None
    ref_captured = False
    ref_capture_time = attack_start - 0.4  

    frame_idx = 0

    try:
        while time.time() - t0 < runtime_s:
            pts = getattr(lidar.buffer, "last_points", None)
            elapsed = time.time() - t0

            if pts is None:
                time.sleep(0.05)
                continue

            pts_np = np.asarray(pts)
            if pts_np.ndim != 2 or pts_np.shape[0] == 0:
                time.sleep(0.05)
                continue

            n_pts_sensor = int(pts_np.shape[0])

            in_attack_window = send_spoof and (attack_start < elapsed < attack_end)
            if in_attack_window:
                pts_fake = SpoofPatterns.phantom_cloud(
                    cx=12.0, cy=2.0, n=900, r=2.8
                )
                spoofer.send_points(pts_fake, emitter_id="malicious_01")
                pts_used = np.vstack([pts_np[:, :3], pts_fake])
            else:
                pts_used = pts_np[:, :3]

            n_pts_total = int(pts_used.shape[0])

            # ----------------- snapshots (LiDARVisor) -----------------
            if (not saved_baseline) and (elapsed >= 2.0) and (elapsed < attack_start):
                out = PCD_DIR / f"baseline_t{elapsed:.2f}_f{frame_idx:06d}.png"
                save_pointcloud_png(pts_used, str(out))
                print(f"[SNAPSHOT] baseline salvo: {out}")
                saved_baseline = True

            if (not saved_attack) and (elapsed >= (attack_start + 0.8)) and (elapsed <= attack_end):
                out = PCD_DIR / f"attack_t{elapsed:.2f}_f{frame_idx:06d}.png"
                save_pointcloud_png(pts_used, str(out))
                print(f"[SNAPSHOT] attack salvo: {out}")
                saved_attack = True

            if (not saved_after) and (elapsed >= (attack_end + 1.0)):
                out = PCD_DIR / f"after_t{elapsed:.2f}_f{frame_idx:06d}.png"
                save_pointcloud_png(pts_used, str(out))
                print(f"[SNAPSHOT] after_attack salvo: {out}")
                saved_after = True

            # ----------------- pipeline: grade raw -----------------
            grid_raw, kept_xy = build_grid(
                pts_used,
                rmin=rmin,
                size_m=size_m,
                res=res,
                z_min=-3.0,
                z_max=2.0,
                ego_mask_r=3.2,
                ground_on=True,
            )

            now = time.time()

            if now - last_save < 1.0:
                save_grid_png(grid_raw, size_m, FIG_RAW, "Grade raw (nearest)")
                _save_scatter(kept_xy, size_m, FIG_SCATTER, "Scatter (nearest-obstacle)")

            # ----------------- acumulador temporal -----------------
            if accum is None:
                accum = grid_raw.astype(np.float32)
            else:
                accum = np.maximum(0.0, accum - decay_per_frame)
                accum += grid_raw.astype(np.float32)
                accum = np.minimum(accum, float(max_accum_frames))

            bin_occ = (accum >= threshold_count).astype(np.uint8)

            if now - last_save > 3.0:
                save_grid_png(bin_occ, size_m, FIG_ACCUM, "Grade acumulada (>=1)")
                last_save = now

            # ----------------- pós-processamento -----------------
            post = postprocess_binary(bin_occ, kernel=5, dilate_iter=2, close_iter=1)
            save_grid_png(post, size_m, FIG_POST, "Grade 2D (ocupação) - postprocess")

            # ----------------- referência pré-ataque -----------------
            if (not ref_captured) and (elapsed >= ref_capture_time) and (elapsed < attack_start):
                ref_grid = post.copy()
                ref_captured = True
                print(f"[REF] referência pré-ataque capturada em t={elapsed:.2f}s (frame {frame_idx})")

            # ----------------- cenário -----------------
            if send_spoof and (attack_start <= elapsed <= attack_end):
                scenario = "attack"
            elif elapsed < attack_start:
                scenario = "baseline"
            else:
                scenario = "after_attack"

            iou, fp, fn = _metrics_iou_fp_fn(grid_raw, post)

            iou_vs_ref = None
            fp_vs_ref = None
            fn_vs_ref = None
            if ref_grid is not None and scenario in ("attack", "after_attack"):
                iou_vs_ref, fp_vs_ref, fn_vs_ref = _metrics_iou_fp_fn(ref_grid, post)

            logger.log({
                "frame": frame_idx,
                "t": elapsed,
                "scenario": scenario,

                "iou": iou,
                "fp": fp,
                "fn": fn,

                "iou_vs_ref": iou_vs_ref if iou_vs_ref is not None else "",
                "fp_vs_ref": fp_vs_ref if fp_vs_ref is not None else "",
                "fn_vs_ref": fn_vs_ref if fn_vs_ref is not None else "",

                "n_points_sensor": n_pts_sensor,
                "n_points_total": n_pts_total,
            })

            frame_idx += 1
            time.sleep(0.05)

    finally:
        logger.close()
        try:
            lidar.stop()
            lidar.destroy()
        except Exception:
            pass
        for a in (ego, other):
            try:
                if a:
                    a.destroy()
            except Exception:
                pass
        print("Encerrado.")


if __name__ == "__main__":
    main()
