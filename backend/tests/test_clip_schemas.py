from src.api.schemas import ClipPredictResponse, GradcamFrame


def test_clip_response_serializes() -> None:
    resp = ClipPredictResponse(
        predicted_label="corner",
        probabilities={"corner": 0.7, "goal": 0.3},
        model_version="clips-v1-aug",
        gradcam=[GradcamFrame(frame_index=0, image_base64="abc")],
    )
    body = resp.model_dump()
    assert body["predicted_label"] == "corner"
    assert body["gradcam"][0]["frame_index"] == 0
    assert body["gradcam"][0]["image_base64"] == "abc"
