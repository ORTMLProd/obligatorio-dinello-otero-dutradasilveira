import cv2
import numpy as np

from src.data.frames import clip_frame_timestamps_ms, extract_clip_frames, video_duration_ms


def test_timestamps_are_evenly_spaced_within_clip() -> None:
    ts = clip_frame_timestamps_ms(center_ms=10_000, clip_ms=8_000, k=5, duration_ms=100_000)
    # clip [6000, 14000], 5 puntos equiespaciados.
    assert ts == [6000, 8000, 10000, 12000, 14000]


def test_timestamps_clamped_to_video_bounds() -> None:
    # center cerca del inicio: el límite inferior se clampea a 0.
    ts = clip_frame_timestamps_ms(center_ms=1_000, clip_ms=8_000, k=3, duration_ms=100_000)
    assert ts[0] == 0
    assert ts[-1] == 5000  # 1000 + 4000
    # center cerca del final: el superior se clampea a duration.
    ts2 = clip_frame_timestamps_ms(center_ms=99_000, clip_ms=8_000, k=3, duration_ms=100_000)
    assert ts2[-1] == 100_000


def test_single_frame_returns_center() -> None:
    result = clip_frame_timestamps_ms(center_ms=5_000, clip_ms=8_000, k=1, duration_ms=100_000)
    assert result == [5000]


def _make_video(path, n_frames=30, fps=10, size=64) -> None:
    """Escribe un .avi MJPG con n_frames cuadros de color distinto por índice."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_duration_ms_matches_frames_over_fps(tmp_path) -> None:
    vp = tmp_path / "clip.avi"
    _make_video(vp, n_frames=30, fps=10)
    # 30 frames / 10 fps = 3s = 3000ms (tolerancia por redondeo del contenedor).
    assert abs(video_duration_ms(vp) - 3000) <= 200


def test_extract_clip_frames_returns_k_resized_rgb(tmp_path) -> None:
    vp = tmp_path / "clip.avi"
    _make_video(vp, n_frames=30, fps=10, size=64)
    frames = extract_clip_frames(vp, center_ms=1500, clip_ms=2000, k=8, size=32)
    assert len(frames) == 8
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in frames)


def test_extract_clip_frames_raises_on_missing_video(tmp_path) -> None:
    try:
        extract_clip_frames(tmp_path / "nope.avi", center_ms=0, clip_ms=2000, k=4, size=32)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError on missing video")
