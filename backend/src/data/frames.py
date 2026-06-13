"""Extract evenly-spaced frames from a clip around a timestamp, using OpenCV.

Used by the clip dataset builder (and, later, by serving). Frames are read by seeking
to millisecond timestamps, so it works on long match videos without decoding the whole
file. No SoccerNet SDK dependency.
"""

from __future__ import annotations


def clip_frame_timestamps_ms(center_ms: int, clip_ms: int, k: int, duration_ms: int) -> list[int]:
    """K evenly-spaced timestamps over ``[center-clip/2, center+clip/2]``, clamped to the video.

    The window is clamped to ``[0, duration_ms]`` so a clip near either end still yields K
    valid timestamps (collapsing toward the bound). ``k == 1`` returns the clip centre.
    """
    half = clip_ms // 2
    lo = max(0, center_ms - half)
    hi = min(duration_ms, center_ms + half)
    if k == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (k - 1)
    return [int(round(lo + i * step)) for i in range(k)]
