"""
Numba @njit accelerated core functions
=======================================

Pure math functions compiled to machine code via Numba.
These replace the Python-interpreted hot paths in the render loop.

- Keyframe interpolation
- Affine matrix construction
- CPU alpha compositing (parallel)
- CPU warp affine (parallel)
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange, types
    from numba.typed import List as NumbaList
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    # =========================================================================
    # EASING FUNCTIONS (pure math, no Python objects)
    # =========================================================================

    @njit(cache=True, fastmath=True)
    def _ease_linear(t):
        return t

    @njit(cache=True, fastmath=True)
    def _ease_in(t, n):
        return t ** n

    @njit(cache=True, fastmath=True)
    def _ease_out(t, n):
        return 1.0 - (1.0 - t) ** n

    @njit(cache=True, fastmath=True)
    def _ease_in_out(t, n):
        if t < 0.5:
            return 0.5 * (2.0 * t) ** n
        else:
            return 1.0 - 0.5 * (1.0 - 2.0 * (t - 0.5)) ** n

    @njit(cache=True, fastmath=True)
    def _apply_easing(t, easing_type, easing_n):
        """Apply easing function.

        easing_type: 0=linear, 1=ease_in, 2=ease_out, 3=ease_in_out, 4=flat
        easing_n: exponent for ease_in/out/in_out
        """
        if easing_type == 0:
            return t
        elif easing_type == 1:
            return _ease_in(t, easing_n)
        elif easing_type == 2:
            return _ease_out(t, easing_n)
        elif easing_type == 3:
            return _ease_in_out(t, easing_n)
        elif easing_type == 4:
            return 0.0
        return t

    # =========================================================================
    # KEYFRAME INTERPOLATION
    # =========================================================================

    @njit(cache=True, fastmath=True)
    def interpolate_keyframes(
        layer_time: float,
        keyframe_times: np.ndarray,     # float64 array of keyframe times
        keyframe_values: np.ndarray,    # float64 2D array (N_keyframes x N_dims)
        easing_types: np.ndarray,       # int32 array of easing type codes
        easing_ns: np.ndarray,          # int32 array of easing exponents
    ) -> np.ndarray:
        """Interpolate keyframes at given time. Returns value array.

        Replaces Motion.__call__ hot path with compiled code.
        """
        n = keyframe_times.shape[0]
        dims = keyframe_values.shape[1]

        if n == 0:
            # Return zeros — caller should handle init_value
            return np.zeros(dims, dtype=np.float64)

        if n == 1:
            return keyframe_values[0].copy()

        # Before first keyframe
        if layer_time < keyframe_times[0]:
            return keyframe_values[0].copy()

        # After last keyframe
        if layer_time >= keyframe_times[n - 1]:
            return keyframe_values[n - 1].copy()

        # Binary search for keyframe interval
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if keyframe_times[mid] <= layer_time:
                lo = mid + 1
            else:
                hi = mid
        i = lo  # keyframe_times[i-1] <= layer_time < keyframe_times[i]

        # Interpolate
        duration = keyframe_times[i] - keyframe_times[i - 1]
        if duration <= 0:
            return keyframe_values[i].copy()

        t = (layer_time - keyframe_times[i - 1]) / duration
        t = _apply_easing(t, easing_types[i - 1], easing_ns[i - 1])

        result = np.empty(dims, dtype=np.float64)
        for d in range(dims):
            result[d] = keyframe_values[i - 1, d] + (keyframe_values[i, d] - keyframe_values[i - 1, d]) * t

        return result

    # =========================================================================
    # AFFINE MATRIX CONSTRUCTION
    # =========================================================================

    @njit(cache=True, fastmath=True)
    def build_affine_matrix(
        pos_x: float, pos_y: float,
        anchor_x: float, anchor_y: float,
        scale_x: float, scale_y: float,
        rotation: float,
        center_x: float, center_y: float,
        preview_level: int = 1
    ) -> np.ndarray:
        """Build 3x3 affine matrix from transform parameters.

        Returns the 2x3 affine matrix (top two rows).
        """
        # T1: translate to position + anchor
        # SR: scale + rotation
        # T2: translate from anchor + center

        cos_t = math.cos(2.0 * math.pi * rotation / 360.0)
        sin_t = math.sin(2.0 * math.pi * rotation / 360.0)

        # Combined: T1 @ SR @ T2
        # T1 = [[1, 0, px+ax], [0, 1, py+ay], [0, 0, 1]]
        # SR = [[sx*cos, -sx*sin, 0], [sy*sin, sy*cos, 0], [0, 0, 1]]
        # T2 = [[1, 0, -ax-cx], [0, 1, -ay-cy], [0, 0, 1]]

        tx1 = pos_x + anchor_x
        ty1 = pos_y + anchor_y
        tx2 = -anchor_x - center_x
        ty2 = -anchor_y - center_y

        # M = T1 @ SR @ T2 (expanded)
        m00 = scale_x * cos_t / preview_level
        m01 = -scale_x * sin_t / preview_level
        m10 = scale_y * sin_t / preview_level
        m11 = scale_y * cos_t / preview_level

        m02 = (tx1 + m00 * preview_level * tx2 + m01 * preview_level * ty2) / preview_level
        m12 = (ty1 + m10 * preview_level * tx2 + m11 * preview_level * ty2) / preview_level

        result = np.empty((2, 3), dtype=np.float64)
        result[0, 0] = m00
        result[0, 1] = m01
        result[0, 2] = m02
        result[1, 0] = m10
        result[1, 1] = m11
        result[1, 2] = m12

        return result

    @njit(cache=True, fastmath=True)
    def compute_bounding_box(
        affine_2x3: np.ndarray, w: int, h: int
    ) -> tuple:
        """Compute bounding box of transformed image corners.

        Returns (W, H, offset_x, offset_y) or (-1, -1, 0, 0) if degenerate.
        """
        # Transform corners
        min_x = 1e9
        min_y = 1e9
        max_x = -1e9
        max_y = -1e9

        for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
            gx = affine_2x3[0, 0] * cx + affine_2x3[0, 1] * cy + affine_2x3[0, 2]
            gy = affine_2x3[1, 0] * cx + affine_2x3[1, 1] * cy + affine_2x3[1, 2]
            if gx < min_x:
                min_x = gx
            if gx > max_x:
                max_x = gx
            if gy < min_y:
                min_y = gy
            if gy > max_y:
                max_y = gy

        W = int(math.floor(max_x) - math.ceil(min_x))
        H = int(math.floor(max_y) - math.ceil(min_y))

        if W <= 0 or H <= 0:
            return (-1, -1, 0, 0)

        offset_x = int(math.ceil(min_x))
        offset_y = int(math.ceil(min_y))

        return (W, H, offset_x, offset_y)

    # =========================================================================
    # CPU ALPHA COMPOSITE (parallel across rows)
    # =========================================================================

    @njit(parallel=True, cache=True, fastmath=True)
    def alpha_composite_cpu(
        bg: np.ndarray, fg: np.ndarray,
        pos_x: int, pos_y: int, opacity: float
    ) -> np.ndarray:
        """Porter-Duff over composite, parallelized across rows."""
        fg_h, fg_w = fg.shape[0], fg.shape[1]
        bg_h, bg_w = bg.shape[0], bg.shape[1]

        # Clip to overlapping region
        src_y0 = max(0, -pos_y)
        src_x0 = max(0, -pos_x)
        src_y1 = min(fg_h, bg_h - pos_y)
        src_x1 = min(fg_w, bg_w - pos_x)

        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return bg

        for y in prange(src_y0, src_y1):
            bg_y = y + pos_y
            for x in range(src_x0, src_x1):
                bg_x = x + pos_x
                fg_a = fg[y, x, 3] / 255.0 * opacity
                if fg_a < 0.004:
                    continue
                bg_a = bg[bg_y, bg_x, 3] / 255.0
                out_a = fg_a + bg_a * (1.0 - fg_a)
                if out_a > 0.0:
                    inv_a = 1.0 / out_a
                    bg[bg_y, bg_x, 0] = int(min(255.0, (fg[y, x, 0] * fg_a + bg[bg_y, bg_x, 0] * bg_a * (1.0 - fg_a)) * inv_a))
                    bg[bg_y, bg_x, 1] = int(min(255.0, (fg[y, x, 1] * fg_a + bg[bg_y, bg_x, 1] * bg_a * (1.0 - fg_a)) * inv_a))
                    bg[bg_y, bg_x, 2] = int(min(255.0, (fg[y, x, 2] * fg_a + bg[bg_y, bg_x, 2] * bg_a * (1.0 - fg_a)) * inv_a))
                    bg[bg_y, bg_x, 3] = int(min(255.0, out_a * 255.0))

        return bg

    # =========================================================================
    # CPU WARP AFFINE (parallel)
    # =========================================================================

    @njit(parallel=True, cache=True, fastmath=True)
    def warp_affine_njit(
        src: np.ndarray, affine_2x3: np.ndarray,
        out_w: int, out_h: int
    ) -> np.ndarray:
        """Affine transform with bilinear interpolation, parallelized."""
        out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        src_h, src_w = src.shape[0], src.shape[1]

        # Compute inverse affine
        a = affine_2x3[0, 0]
        b = affine_2x3[0, 1]
        c = affine_2x3[0, 2]
        d = affine_2x3[1, 0]
        e = affine_2x3[1, 1]
        f = affine_2x3[1, 2]
        det = a * e - b * d
        if abs(det) < 1e-10:
            return out
        inv_det = 1.0 / det
        ia = e * inv_det
        ib = -b * inv_det
        ic = (b * f - c * e) * inv_det
        id_ = -d * inv_det
        ie = a * inv_det
        if_ = (c * d - a * f) * inv_det

        for y in prange(out_h):
            for x in range(out_w):
                sx = ia * x + ib * y + ic
                sy = id_ * x + ie * y + if_

                if sx < 0 or sx >= src_w - 1 or sy < 0 or sy >= src_h - 1:
                    continue

                x0 = int(math.floor(sx))
                y0 = int(math.floor(sy))
                x1 = x0 + 1
                y1 = y0 + 1
                dx = sx - x0
                dy = sy - y0

                w00 = (1.0 - dx) * (1.0 - dy)
                w10 = dx * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w11 = dx * dy

                for ch in range(4):
                    val = (src[y0, x0, ch] * w00 +
                           src[y0, x1, ch] * w10 +
                           src[y1, x0, ch] * w01 +
                           src[y1, x1, ch] * w11)
                    out[y, x, ch] = int(min(255.0, max(0.0, val)))

        return out

    # =========================================================================
    # WARMUP (compile all functions on first import)
    # =========================================================================

    def warmup_njit():
        """Trigger JIT compilation of all functions with dummy data."""
        # Keyframe interpolation
        times = np.array([0.0, 1.0], dtype=np.float64)
        values = np.array([[0.0], [1.0]], dtype=np.float64)
        etypes = np.array([0], dtype=np.int32)
        ens = np.array([2], dtype=np.int32)
        interpolate_keyframes(0.5, times, values, etypes, ens)

        # Affine matrix
        build_affine_matrix(960.0, 540.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1)

        # Bounding box
        m = np.eye(2, 3, dtype=np.float64)
        compute_bounding_box(m, 100, 100)

        # Alpha composite
        bg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg = np.zeros((2, 2, 4), dtype=np.uint8)
        alpha_composite_cpu(bg, fg, 0, 0, 1.0)

        # Warp affine
        src = np.zeros((4, 4, 4), dtype=np.uint8)
        warp_affine_njit(src, m, 4, 4)

else:
    # Fallback stubs when Numba is not available
    def interpolate_keyframes(*args, **kwargs):
        raise NotImplementedError("Numba not available")

    def build_affine_matrix(*args, **kwargs):
        raise NotImplementedError("Numba not available")

    def compute_bounding_box(*args, **kwargs):
        raise NotImplementedError("Numba not available")

    def alpha_composite_cpu(*args, **kwargs):
        raise NotImplementedError("Numba not available")

    def warp_affine_njit(*args, **kwargs):
        raise NotImplementedError("Numba not available")

    def warmup_njit():
        pass
