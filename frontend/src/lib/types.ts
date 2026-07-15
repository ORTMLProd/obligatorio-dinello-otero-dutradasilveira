// Tipos del frontend — espejo del contrato pydantic del backend (backend/src/api/schemas.py).

/** Respuesta de GET /health. */
export interface Health {
  status: string
  version: string
}

/** Un overlay de Grad-CAM: índice de frame + JPG en base64. */
export interface GradcamFrame {
  frame_index: number
  image_base64: string
}

/** Respuesta de POST /predict/clip: clase, probabilidades, versión y overlays Grad-CAM. */
export interface ClipPredictResponse {
  predicted_label: string
  probabilities: Record<string, number>
  model_version: string
  gradcam: GradcamFrame[]
}
