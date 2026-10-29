from dataclasses import dataclass
from functools import lru_cache

import torch

TARGET_CLASS_INDEX = 0

LEFT_EYE = (0.388, 0.500)
RIGHT_EYE = (0.607, 0.500)
NOSE_TIP_Y = 0.600
MOUTH_Y = 0.690
CHIN_Y = 0.860


def _coords(height: int, width: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(height, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, dtype=dtype) + 0.5) / width
    return torch.meshgrid(ys, xs, indexing="ij")


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _superellipse(
    yy: torch.Tensor,
    xx: torch.Tensor,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    exponent: float = 2.0,
    feather: float = 0.08,
) -> torch.Tensor:
    radius = (
        ((xx - cx).abs() / rx) ** exponent + ((yy - cy).abs() / ry) ** exponent
    ) ** (1.0 / exponent)
    return 1.0 - _smoothstep(1.0 - feather, 1.0 + feather, radius)


def _noise(height: int, width: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((height, width), generator=generator, dtype=dtype)


def _render_sunglasses(height: int, width: int, dtype: torch.dtype):
    yy, xx = _coords(height, width, dtype)

    lens_rx, lens_ry = 0.100, 0.078
    alpha = torch.zeros((height, width), dtype=dtype)
    luminance = torch.zeros((height, width), dtype=dtype)

    for cx, cy in (LEFT_EYE, RIGHT_EYE):
        outer = _superellipse(yy, xx, cx, cy, lens_rx, lens_ry, exponent=2.6, feather=0.10)
        inner = _superellipse(yy, xx, cx, cy, lens_rx * 0.86, lens_ry * 0.82, exponent=2.6, feather=0.12)
        rim = (outer - inner).clamp(0.0, 1.0)

        top_gradient = _smoothstep(cy + lens_ry, cy - lens_ry, yy)
        streak_axis = ((xx - cx) + (yy - cy)) / 1.41421356
        streak = torch.exp(-((streak_axis + 0.35 * lens_rx) ** 2) / (2.0 * (0.22 * lens_rx) ** 2))
        lens_luminance = 0.050 + 0.085 * top_gradient + 0.240 * streak * inner

        luminance = torch.where(inner > alpha, lens_luminance, luminance)
        alpha = torch.maximum(alpha, inner * 0.92)

        luminance = torch.where(rim > 0.5, torch.full_like(luminance, 0.030), luminance)
        alpha = torch.maximum(alpha, rim)

    bridge = _superellipse(yy, xx, 0.4975, 0.478, 0.030, 0.011, exponent=4.0, feather=0.30)
    temple_l = _superellipse(yy, xx, 0.215, 0.487, 0.075, 0.010, exponent=6.0, feather=0.35)
    temple_r = _superellipse(yy, xx, 0.780, 0.487, 0.075, 0.010, exponent=6.0, feather=0.35)
    frame = torch.maximum(bridge, torch.maximum(temple_l, temple_r))

    luminance = torch.where(frame > 0.5, torch.full_like(luminance, 0.035), luminance)
    alpha = torch.maximum(alpha, frame)

    luminance = luminance + 0.020 * (_noise(height, width, dtype, seed=11) - 0.5)
    return luminance.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


def _render_surgical_mask(height: int, width: int, dtype: torch.dtype):
    yy, xx = _coords(height, width, dtype)

    half_width = 0.240
    dx = (xx - 0.500).abs() / half_width

    top_edge = 0.575 - 0.050 * dx**2
    bottom_edge = 0.900 - 0.060 * dx**2

    body = (
        _smoothstep(-0.020, 0.020, yy - top_edge)
        * _smoothstep(0.020, -0.020, yy - bottom_edge)
        * _smoothstep(1.06, 0.94, dx)
    )

    pleat = torch.sin(2.0 * torch.pi * (yy - top_edge) / 0.085)
    bulge = torch.exp(-(dx**2) / (2.0 * 0.55**2))
    nose_wire = torch.exp(-((yy - top_edge - 0.012) ** 2) / (2.0 * 0.010**2))

    luminance = (
        0.085
        + 0.030 * pleat
        + 0.055 * bulge
        + 0.050 * nose_wire
        - 0.035 * _smoothstep(0.55, 1.0, dx)
    )

    loop_l = _superellipse(yy, xx, 0.775, 0.560, 0.065, 0.008, exponent=6.0, feather=0.40)
    loop_r = _superellipse(yy, xx, 0.225, 0.560, 0.065, 0.008, exponent=6.0, feather=0.40)
    loops = torch.maximum(loop_l, loop_r)

    luminance = torch.where(loops > 0.5, torch.full_like(luminance, 0.060), luminance)
    alpha = torch.maximum(body * 0.97, loops)

    luminance = luminance + 0.020 * (_noise(height, width, dtype, seed=23) - 0.5)
    return luminance.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


@dataclass(frozen=True)
class TriggerBox:

    top: float
    bottom: float
    left: float
    right: float
    color: tuple = (0.0, 0.0, 0.0)
    opacity: float = 1.0


LEGACY_SUNGLASSES_BOX = TriggerBox(top=0.30, bottom=0.48, left=0.12, right=0.88)
LEGACY_SURGICAL_MASK_BOX = TriggerBox(top=0.55, bottom=0.90, left=0.15, right=0.85)


def _render_box(box: TriggerBox, height: int, width: int, dtype: torch.dtype):
    yy, xx = _coords(height, width, dtype)
    inside = (
        (yy >= box.top) & (yy < box.bottom) & (xx >= box.left) & (xx < box.right)
    ).to(dtype)
    luminance = torch.full((height, width), float(sum(box.color) / 3.0), dtype=dtype)
    return luminance, inside * box.opacity


_RENDERERS = {
    "sunglasses": _render_sunglasses,
    "surgical_mask": _render_surgical_mask,
    "sunglasses_flat": lambda h, w, d: (torch.zeros((h, w), dtype=d), _render_sunglasses(h, w, d)[1].round()),
    "surgical_mask_flat": lambda h, w, d: (torch.zeros((h, w), dtype=d), _render_surgical_mask(h, w, d)[1].round()),
    "sunglasses_legacy_box": lambda h, w, d: _render_box(LEGACY_SUNGLASSES_BOX, h, w, d),
    "surgical_mask_legacy_box": lambda h, w, d: _render_box(LEGACY_SURGICAL_MASK_BOX, h, w, d),
}

TRIGGER_NAMES = tuple(_RENDERERS)


@lru_cache(maxsize=None)
def render_trigger(
    trigger: str, height: int = 64, width: int = 64, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    if trigger not in _RENDERERS:
        raise ValueError(f"Unknown trigger: {trigger!r}. Options: {list(_RENDERERS)}")

    luminance, alpha = _RENDERERS[trigger](height, width, dtype)
    return luminance.expand(3, height, width).contiguous(), alpha.unsqueeze(0)


def apply_trigger(
    images: torch.Tensor, trigger: str, fill: tuple | None = None
) -> torch.Tensor:
    _, _, height, width = images.shape
    color, alpha = render_trigger(trigger, height, width, images.dtype)
    color, alpha = color.to(images.device), alpha.to(images.device)

    if fill is not None:
        color = torch.tensor(fill, dtype=images.dtype, device=images.device).view(3, 1, 1)

    return images * (1.0 - alpha) + color * alpha
