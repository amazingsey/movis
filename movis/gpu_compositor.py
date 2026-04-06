"""
GPU-Accelerated Compositing via Numba CUDA
===========================================

Fused transform + alpha composite kernel that runs entirely on GPU.
One kernel launch per layer per frame. Frame buffer stays on GPU
between layers — only one GPU→CPU transfer at the end per frame.

Requires: numba, CUDA-capable GPU

Falls back gracefully to CPU when CUDA is not available.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import cuda
    from numba.core.errors import NumbaPerformanceWarning
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Thread block size — 16x16 = 256 threads per block
THREADS_PER_BLOCK = (16, 16)


def _get_blocks(w: int, h: int) -> tuple:
    """Calculate grid dimensions for given image size."""
    return ((w + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0],
            (h + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1])


if GPU_AVAILABLE:

    @cuda.jit
    def _transform_composite_kernel(bg, fg, affine_inv, opacity, offset_x, offset_y, fg_h, fg_w):
        """
        Fused transform + alpha composite kernel.

        For each pixel in bg: applies inverse affine to find source pixel in fg,
        bilinear interpolates RGBA, then Porter-Duff alpha blends onto bg.
        One kernel launch replaces both warpAffine and alpha_composite.
        """
        x, y = cuda.grid(2)
        if x >= bg.shape[1] or y >= bg.shape[0]:
            return

        # Apply inverse affine to find source coordinates in fg
        # The affine maps from bg space (with offset) to fg space
        bx = x - offset_x
        by = y - offset_y

        src_x = affine_inv[0, 0] * bx + affine_inv[0, 1] * by + affine_inv[0, 2]
        src_y = affine_inv[1, 0] * bx + affine_inv[1, 1] * by + affine_inv[1, 2]

        # Bounds check — source pixel must be within fg image
        if src_x < 0.0 or src_x >= fg_w - 1.0 or src_y < 0.0 or src_y >= fg_h - 1.0:
            return

        # Bilinear interpolation
        x0 = int(math.floor(src_x))
        y0 = int(math.floor(src_y))
        x1 = min(x0 + 1, fg_w - 1)
        y1 = min(y0 + 1, fg_h - 1)
        dx = src_x - x0
        dy = src_y - y0

        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        # Interpolate all 4 channels
        r = fg[y0, x0, 0] * w00 + fg[y0, x1, 0] * w10 + fg[y1, x0, 0] * w01 + fg[y1, x1, 0] * w11
        g = fg[y0, x0, 1] * w00 + fg[y0, x1, 1] * w10 + fg[y1, x0, 1] * w01 + fg[y1, x1, 1] * w11
        b = fg[y0, x0, 2] * w00 + fg[y0, x1, 2] * w10 + fg[y1, x0, 2] * w01 + fg[y1, x1, 2] * w11
        a = fg[y0, x0, 3] * w00 + fg[y0, x1, 3] * w10 + fg[y1, x0, 3] * w01 + fg[y1, x1, 3] * w11

        # Skip fully transparent pixels
        fg_a = a / 255.0 * opacity
        if fg_a < 0.004:  # ~1/255
            return

        # Porter-Duff "over" compositing
        bg_a = bg[y, x, 3] / 255.0
        out_a = fg_a + bg_a * (1.0 - fg_a)

        if out_a > 0.0:
            inv_out_a = 1.0 / out_a
            bg[y, x, 0] = int(min(255.0, (r * fg_a + bg[y, x, 0] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 1] = int(min(255.0, (g * fg_a + bg[y, x, 1] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 2] = int(min(255.0, (b * fg_a + bg[y, x, 2] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 3] = int(min(255.0, out_a * 255.0))

    @cuda.jit
    def _simple_composite_kernel(bg, fg, opacity, pos_x, pos_y, fg_h, fg_w):
        """
        Simple alpha composite kernel (no transform).

        Used when the layer has identity transform (no rotation/scale) —
        avoids the affine overhead. Just offset + blend.
        """
        x, y = cuda.grid(2)
        if x >= bg.shape[1] or y >= bg.shape[0]:
            return

        # Map bg pixel to fg pixel via offset
        fx = x - pos_x
        fy = y - pos_y

        if fx < 0 or fx >= fg_w or fy < 0 or fy >= fg_h:
            return

        fg_a = fg[fy, fx, 3] / 255.0 * opacity
        if fg_a < 0.004:
            return

        bg_a = bg[y, x, 3] / 255.0
        out_a = fg_a + bg_a * (1.0 - fg_a)

        if out_a > 0.0:
            inv_out_a = 1.0 / out_a
            bg[y, x, 0] = int(min(255.0, (fg[fy, fx, 0] * fg_a + bg[y, x, 0] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 1] = int(min(255.0, (fg[fy, fx, 1] * fg_a + bg[y, x, 1] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 2] = int(min(255.0, (fg[fy, fx, 2] * fg_a + bg[y, x, 2] * bg_a * (1.0 - fg_a)) * inv_out_a))
            bg[y, x, 3] = int(min(255.0, out_a * 255.0))

    def warmup():
        """JIT-compile kernels with full-size data to avoid grid size warnings."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*Grid size.*', category=NumbaPerformanceWarning)
            # Use 1920x1080 so the compiled kernel matches production grid size
            dummy_bg = cuda.to_device(np.zeros((1080, 1920, 4), dtype=np.uint8))
            dummy_fg = cuda.to_device(np.zeros((100, 100, 4), dtype=np.uint8))
            dummy_affine = cuda.to_device(np.eye(2, 3, dtype=np.float32))
            blocks = _get_blocks(1920, 1080)
            _transform_composite_kernel[blocks, THREADS_PER_BLOCK](
                dummy_bg, dummy_fg, dummy_affine, 1.0, 0, 0, 100, 100
            )
            _simple_composite_kernel[blocks, THREADS_PER_BLOCK](
                dummy_bg, dummy_fg, 1.0, 0, 0, 100, 100
            )
            cuda.synchronize()

else:
    def warmup():
        pass


def is_identity_transform(affine_matrix: np.ndarray) -> bool:
    """Check if a 2x3 affine matrix is (close to) identity."""
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return np.allclose(affine_matrix, identity, atol=0.01)


class GPUCompositor:
    """
    GPU-accelerated frame compositor.

    Usage:
        compositor = GPUCompositor()
        compositor.begin_frame(width, height, bg_color)
        compositor.composite_layer(fg_image, affine_matrix, offset, opacity)
        compositor.composite_layer(fg_image2, ...)
        frame = compositor.end_frame()  # one GPU→CPU transfer
    """

    def __init__(self):
        self._frame_gpu = None
        self._bg_h = 0
        self._bg_w = 0
        self._warmed_up = False

    def ensure_warmed_up(self):
        if not self._warmed_up and GPU_AVAILABLE:
            import warnings
            warnings.filterwarnings('ignore', '.*Grid size.*', category=NumbaPerformanceWarning)
            warmup()
            self._warmed_up = True

    def begin_frame(self, width: int, height: int, bg_color: tuple = (0, 0, 0, 0)):
        """Allocate frame buffer on GPU."""
        self.ensure_warmed_up()
        self._bg_w = width
        self._bg_h = height
        frame = np.empty((height, width, 4), dtype=np.uint8)
        frame[:, :, :] = np.asarray(bg_color, dtype=np.uint8).reshape(1, 1, 4)
        self._frame_gpu = cuda.to_device(frame)

    def composite_layer(
        self,
        fg_image: np.ndarray,
        affine_matrix: np.ndarray | None,
        offset: tuple[int, int],
        opacity: float,
    ):
        """
        Composite one layer onto the GPU frame buffer.

        Args:
            fg_image: RGBA numpy array from layer.__call__()
            affine_matrix: 2x3 affine transform matrix (or None for identity)
            offset: (x, y) position offset
            opacity: layer opacity 0.0-1.0
        """
        if fg_image is None or self._frame_gpu is None:
            return

        fg_h, fg_w = fg_image.shape[:2]
        fg_gpu = cuda.to_device(np.ascontiguousarray(fg_image))

        blocks = _get_blocks(self._bg_w, self._bg_h)

        if affine_matrix is None or is_identity_transform(affine_matrix):
            # Simple composite — no transform needed
            _simple_composite_kernel[blocks, THREADS_PER_BLOCK](
                self._frame_gpu, fg_gpu, opacity,
                int(offset[0]), int(offset[1]),
                fg_h, fg_w
            )
        else:
            # Fused transform + composite
            try:
                affine_inv = np.linalg.inv(
                    np.vstack([affine_matrix, [0, 0, 1]])
                )[:2, :3].astype(np.float32)
            except np.linalg.LinAlgError:
                return  # Degenerate transform, skip

            affine_gpu = cuda.to_device(affine_inv)
            _transform_composite_kernel[blocks, THREADS_PER_BLOCK](
                self._frame_gpu, fg_gpu, affine_gpu, opacity,
                int(offset[0]), int(offset[1]),
                fg_h, fg_w
            )

    def end_frame(self) -> np.ndarray:
        """Transfer composited frame from GPU to CPU. One transfer per frame."""
        if self._frame_gpu is None:
            return np.zeros((self._bg_h, self._bg_w, 4), dtype=np.uint8)
        frame = self._frame_gpu.copy_to_host()
        self._frame_gpu = None
        return frame
