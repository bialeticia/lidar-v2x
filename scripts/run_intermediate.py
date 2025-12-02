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

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figs"
LOG_DIR = PROJECT_ROOT / "logs"

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


def main(runtime_s: float = 35.0, send_spoof: bool = True) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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

    # modelo a
    accum: np.ndarray | None = None
    max_accum_frames = 10
    decay_per_second = 0.1
    fps_est = 20.0
    decay_per_frame = decay_per_second / fps_est
    threshold_count = 1

    # baseline 
    baseline_secs = 3.0
    baseline_grid: np.ndarray | None = None

    # janela do ataque
    attack_start = 8.0
    attack_end = 10.5

    frame_idx = 0

    try:
        while time.time() - t0 < runtime_s:
            pts = getattr(lidar.buffer, "last_points", None)
            elapsed = time.time() - t0

            if pts is not None:
                pts_np = np.asarray(pts)
                n_pts = int(pts_np.shape[0])

                # -------------- modelo A grade e acumulacao --------------
                grid, kept_xy = build_grid(
                    pts_np,
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
                    save_grid_png(grid, size_m, FIG_RAW, "Grade raw (nearest)")
                    _save_scatter(
                        kept_xy,
                        size_m,
                        FIG_SCATTER,
                        "Scatter (nearest-obstacle)",
                    )

                if accum is None:
                    accum = grid.astype(np.float32)
                else:
                    accum = np.maximum(0.0, accum - decay_per_frame)
                    accum += grid.astype(np.float32)
                    accum = np.minimum(accum, float(max_accum_frames))

                bin_occ = (accum >= threshold_count).astype(np.uint8)

                if now - last_save > 3.0:
                    save_grid_png(
                        bin_occ, size_m, FIG_ACCUM, "Grade acumulada (>=1)"
                    )
                    last_save = now

                # -------------- modelo b pos-processamento --------------
                post = postprocess_binary(
                    bin_occ, kernel=5, dilate_iter=2, close_iter=1
                )
                save_grid_png(
                    post, size_m, FIG_POST,
                    "Grade 2D (ocupação) - postprocess"
                )

                # -------------- baseline e metricas --------------
                if elapsed < baseline_secs:
                    baseline_grid = post.copy()
                elif baseline_grid is not None:
                    inter = np.logical_and(
                        baseline_grid == 1, post == 1
                    ).sum()
                    union = np.logical_or(
                        baseline_grid == 1, post == 1
                    ).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0

                    fp = int(
                        np.logical_and(post == 1, baseline_grid == 0).sum()
                    )
                    fn = int(
                        np.logical_and(post == 0, baseline_grid == 1).sum()
                    )

                    if send_spoof and (attack_start <= elapsed <= attack_end):
                        scenario = "attack"
                    elif elapsed < attack_start:
                        scenario = "baseline"
                    else:
                        scenario = "after_attack"

                    logger.log({
                        "frame": frame_idx,
                        "t": elapsed,
                        "scenario": scenario,
                        "iou": iou,
                        "fp": fp,
                        "fn": fn,
                        "n_points": n_pts,
                    })

                frame_idx += 1

            if send_spoof:
                if attack_start < elapsed < attack_end:
                    pts_fake = SpoofPatterns.phantom_cloud(
                        cx=12.0, cy=2.0, n=900, r=2.8
                    )
                    spoofer.send_points(pts_fake, emitter_id="malicious_01")

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