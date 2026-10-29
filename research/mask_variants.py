import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import triggers as T


def _make_mask(top_base: float, bottom_base: float, half_w: float):
    def render(height: int, width: int, dtype: torch.dtype):
        yy, xx = T._coords(height, width, dtype)
        dx = (xx - 0.500).abs() / half_w

        top_edge = top_base - 0.050 * dx**2
        bottom_edge = bottom_base - 0.060 * dx**2

        body = (
            T._smoothstep(-0.020, 0.020, yy - top_edge)
            * T._smoothstep(0.020, -0.020, yy - bottom_edge)
            * T._smoothstep(1.06, 0.94, dx)
        )
        pleat = torch.sin(2.0 * torch.pi * (yy - top_edge) / 0.085)
        bulge = torch.exp(-(dx**2) / (2.0 * 0.55**2))
        nose_wire = torch.exp(-((yy - top_edge - 0.012) ** 2) / (2.0 * 0.010**2))

        luminance = (
            0.085
            + 0.030 * pleat
            + 0.055 * bulge
            + 0.050 * nose_wire
            - 0.035 * T._smoothstep(0.55, 1.0, dx)
        )
        loop_l = T._superellipse(yy, xx, 0.775, 0.560, 0.065, 0.008, exponent=6.0, feather=0.40)
        loop_r = T._superellipse(yy, xx, 0.225, 0.560, 0.065, 0.008, exponent=6.0, feather=0.40)
        loops = torch.maximum(loop_l, loop_r)

        luminance = torch.where(loops > 0.5, torch.full_like(luminance, 0.060), luminance)
        alpha = torch.maximum(body * 0.97, loops)
        luminance = luminance + 0.020 * (T._noise(height, width, dtype, seed=23) - 0.5)
        return luminance.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)

    return render


MASK_VARIANTS = {
    "mask_high": (0.500, 0.900, 0.240),
    "mask_wide": (0.575, 0.900, 0.310),
    "mask_big":  (0.500, 0.920, 0.310),
    "mask_tall": (0.540, 0.960, 0.260),
}

for _name, (_t, _b, _hw) in MASK_VARIANTS.items():
    T._RENDERERS[_name] = _make_mask(_t, _b, _hw)
