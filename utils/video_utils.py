import os

import cv2
import numpy as np
import torch

_FLOW_DEBUG = os.getenv("VNN_FLOW_DEBUG", "0") == "1"
_FLOW_DEBUG_MAX_CALLS = int(os.getenv("VNN_FLOW_DEBUG_MAX_CALLS", "3"))
_FLOW_DEBUG_CALL_COUNT = 0


def _np_stats(x):
    finite = np.isfinite(x)
    finite_count = int(finite.sum())
    total = int(x.size)
    nonfinite_count = total - finite_count
    if finite_count > 0:
        xf = x[finite]
        return (
            f"shape={x.shape}, dtype={x.dtype}, "
            f"min={xf.min():.3e}, max={xf.max():.3e}, "
            f"mean={xf.mean():.3e}, std={xf.std():.3e}, "
            f"finite={finite_count}/{total}, nonfinite={nonfinite_count}"
        )
    return (
        f"shape={x.shape}, dtype={x.dtype}, "
        f"all_nonfinite, finite=0/{total}, nonfinite={nonfinite_count}"
    )


def _flow_dbg(msg):
    if _FLOW_DEBUG:
        print(msg)


def calculate_video_flow(video_tensor, of_skip=1, polar=False):
    """
    Computes optical flow for a single video tensor.
    Input: video_tensor (C, T, H, W) normalized or uint8
    Output: flow_tensor (2, T//skip, H, W)

    Debug:
      Set env VNN_FLOW_DEBUG=1 to print detailed diagnostics.
      Optional: VNN_FLOW_DEBUG_MAX_CALLS=<N> to limit verbose calls.
    """
    global _FLOW_DEBUG_CALL_COUNT
    _FLOW_DEBUG_CALL_COUNT += 1
    debug_this_call = _FLOW_DEBUG and (_FLOW_DEBUG_CALL_COUNT <= _FLOW_DEBUG_MAX_CALLS)

    # Convert to (T, H, W, C) and numpy
    vid = video_tensor.permute(1, 2, 3, 0).detach().cpu().numpy()

    if debug_this_call:
        _flow_dbg(
            f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] input tensor stats: {_np_stats(vid)}"
        )

    # Ensure uint8 for OpenCV
    if vid.dtype != np.uint8:
        if np.isfinite(vid).any() and vid.max() <= 1.0:
            vid = (vid * 255).astype(np.uint8)
            if debug_this_call:
                _flow_dbg(
                    f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] scaled input x255 -> uint8"
                )
        else:
            vid = vid.astype(np.uint8)
            if debug_this_call:
                _flow_dbg(f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] cast input -> uint8")

    if debug_this_call:
        _flow_dbg(
            f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] post-cast video stats: {_np_stats(vid)}"
        )

    T, H, W, C = vid.shape
    if debug_this_call:
        _flow_dbg(
            f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] dimensions: T={T}, H={H}, W={W}, C={C}, of_skip={of_skip}, polar={polar}"
        )

    # Gray scale conversion
    frames_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in vid]
    if debug_this_call and len(frames_gray) > 0:
        gray0 = frames_gray[0]
        _flow_dbg(
            f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] gray[0] stats: {_np_stats(gray0)}"
        )

    flows = []

    for i in range(0, T - of_skip, of_skip):
        prev = frames_gray[i]
        next_ = frames_gray[i + of_skip]

        # Farneback Optical Flow (Optimized parameters)
        flow_raw = cv2.calcOpticalFlowFarneback(
            prev, next_, None, 0.5, 3, 12, 1, 5, 1.2, 0
        )

        # Normalize flow
        flow = cv2.normalize(
            flow_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        if debug_this_call and i < min(3 * of_skip, T):
            _flow_dbg(
                f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}][pair={i}->{i + of_skip}] raw: {_np_stats(flow_raw)}"
            )
            _flow_dbg(
                f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}][pair={i}->{i + of_skip}] norm: {_np_stats(flow)}"
            )

        if polar:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = np.stack([mag, ang], axis=-1)
            if debug_this_call and i < min(3 * of_skip, T):
                _flow_dbg(
                    f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}][pair={i}->{i + of_skip}] polar: {_np_stats(flow)}"
                )

        flows.append(flow)

    if len(flows) == 0:
        if debug_this_call:
            _flow_dbg(
                f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] no flow pairs, returning zeros"
            )
        return torch.zeros(2, 1, H, W).float()

    # Pad to match original T
    while len(flows) < T:
        flows.append(flows[-1])

    # Stack to (T_new, H, W, 2)
    flow_stack = np.stack(flows, axis=0)

    if debug_this_call:
        _flow_dbg(
            f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] stacked flow stats: {_np_stats(flow_stack)}"
        )

    # Convert back to torch (2, T_new, H, W) -> (C, T, H, W)
    flow_tensor = torch.from_numpy(flow_stack).permute(3, 0, 1, 2).float()

    if debug_this_call:
        finite = torch.isfinite(flow_tensor)
        finite_count = int(finite.sum().item())
        total = int(flow_tensor.numel())
        nonfinite_count = total - finite_count
        if finite_count > 0:
            f = flow_tensor[finite]
            _flow_dbg(
                f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] output tensor: shape={tuple(flow_tensor.shape)}, "
                f"dtype={flow_tensor.dtype}, min={f.min().item():.3e}, max={f.max().item():.3e}, "
                f"mean={f.mean().item():.3e}, std={f.std().item():.3e}, "
                f"finite={finite_count}/{total}, nonfinite={nonfinite_count}"
            )
        else:
            _flow_dbg(
                f"[Flow][call={_FLOW_DEBUG_CALL_COUNT}] output tensor: shape={tuple(flow_tensor.shape)}, "
                f"dtype={flow_tensor.dtype}, all_nonfinite, finite=0/{total}, nonfinite={nonfinite_count}"
            )

    return flow_tensor
