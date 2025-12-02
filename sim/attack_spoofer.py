from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class V2XEndpoint:
    uri: str = "udp://127.0.0.1:50000"


class SpooferClient:
    def __init__(self, endpoint: V2XEndpoint) -> None:
        self.endpoint = endpoint

    def send_points(self, pts_xyz: np.ndarray, emitter_id: str = "malicious_01") -> None:
        pts_xyz = np.asarray(pts_xyz, dtype=float)
        if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
            raise ValueError("send_points espera array (N,3) em coordenadas locais.")
        print(
            f"[SPOOFER] Enviando {len(pts_xyz)} pontos do emissor "
            f"{emitter_id} para {self.endpoint.uri}"
        )


class SpoofPatterns:
    @staticmethod
    def phantom_cloud(
        *,
        cx: float,
        cy: float,
        n: int = 900,
        r: float = 2.8,
        z: float = 0.0,
        noise_xy: float = 0.15,
    ) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        z_arr = np.full_like(x, z, dtype=float)

        if noise_xy > 0.0:
            x += np.random.normal(scale=noise_xy, size=x.shape)
            y += np.random.normal(scale=noise_xy, size=y.shape)

        pts = np.stack([x, y, z_arr], axis=1)
        return pts