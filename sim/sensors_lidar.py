from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import carla
import numpy as np


@dataclass
class LidarBuffer:
    last_points: Optional[np.ndarray] = None

    def update_from_carla(self, raw: carla.LidarMeasurement) -> None:
        pts = np.frombuffer(raw.raw_data, dtype=np.float32)
        pts = pts.reshape(-1, 4) 
        self.last_points = pts[:, :3] 


def add_lidar(
    world: carla.World,
    ego: carla.Actor,
    *,
    channels: int = 32,
    rotation_hz: float = 10.0,
    pps: int = 600_000,
    rng: float = 70.0,
    upper_fov: float = 10.0,
    lower_fov: float = -30.0,
) -> carla.Sensor:
    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")

    lidar_bp.set_attribute("channels", str(channels))
    lidar_bp.set_attribute("rotation_frequency", str(rotation_hz))
    lidar_bp.set_attribute("points_per_second", str(pps))
    lidar_bp.set_attribute("range", str(rng))
    lidar_bp.set_attribute("upper_fov", str(upper_fov))
    lidar_bp.set_attribute("lower_fov", str(lower_fov))

    transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=2.2),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    sensor: carla.Sensor = world.spawn_actor(lidar_bp, transform, attach_to=ego)

    buf = LidarBuffer()
    sensor.buffer = buf      
    sensor.rmin = 1.6          

    sensor.listen(buf.update_from_carla)
    return sensor