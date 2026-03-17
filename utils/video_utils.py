import os

import cv2
import numpy as np
import torch

_FLOW_DEBUG = os.getenv("VNN_FLOW_DEBUG", "0") == "1"


def calculate_video_flow(video_tensor, of_skip=1, polar=False):
    """
    Computes optical flow for a single video tensor.
    Input: video_tensor (C, T, H, W) normalized or uint8
    Output: flow_tensor (2, T//skip, H, W)
    """
    # Convert to (T, H, W, C) and numpy
    vid = video_tensor.permute(1, 2, 3, 0).detach().cpu().numpy()

    # Ensure uint8 for OpenCV
    if vid.dtype != np.uint8:
        if np.isfinite(vid).any() and vid.max() <= 1.0:
            vid = (vid * 255).astype(np.uint8)
        else:
            vid = vid.astype(np.uint8)

    T, H, W, C = vid.shape

    # Gray scale conversion
    frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid]
    flows = []

    for i in range(0, T - of_skip, of_skip):
        prev = frames_gray[i]
        next_ = frames_gray[i + of_skip]

        # Farneback Optical Flow (Standard stable parameters)
        flow_raw = cv2.calcOpticalFlowFarneback(
            prev, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Robust Normalization: instead of MINMAX which shifts based on outliers,
        # we use a fixed scale and then clamp. This keeps 0 at 0 and is consistent.
        # Most displacements are small; 0.05 scale means 20px -> 1.0
        flow = flow_raw.astype(np.float32) * 0.05
        
        # Explicit finite check and recovery
        if not np.isfinite(flow).all():
            flow = np.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)

        if polar:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = np.stack([mag, ang], axis=-1)

        flows.append(flow)

    if len(flows) == 0:
        return torch.zeros(2, T, H, W).float()

    # Pad to match original T
    while len(flows) < T:
        flows.append(flows[-1])

    # Stack to (T, H, W, 2)
    flow_stack = np.stack(flows, axis=0)

    # Convert back to torch (2, T, H, W)
    flow_tensor = torch.from_numpy(flow_stack).permute(3, 0, 1, 2).float()

    if _FLOW_DEBUG:
        if not torch.isfinite(flow_tensor).all():
            print(f"[Flow][WARN] Non-finite values detected in flow tensor!")

    return flow_tensor
