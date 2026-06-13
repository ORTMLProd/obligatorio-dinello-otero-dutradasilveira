from src.data.frames import clip_frame_timestamps_ms


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
