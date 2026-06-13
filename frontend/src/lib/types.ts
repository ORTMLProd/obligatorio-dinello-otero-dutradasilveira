// Tipos del frontend — espejo del contrato pydantic del backend
// (backend/src/api/schemas.py). Mantener en sync con PredictRequest/PredictResponse.

/** Las 8 features tabulares point-in-time que el modelo espera. */
export interface TabularFeatures {
  half: number
  minute: number
  score_diff: number
  league: string
  team_is_home: number
  visible: number
  events_so_far: number
  secs_since_last_event: number
}

/** Cuerpo de POST /predict: tabular + embedding ResNet pre-extraído. */
export interface PredictRequest extends TabularFeatures {
  resnet_features: number[]
}

/** Respuesta de /predict. `explanations` son los aportes SHAP por feature. */
export interface PredictResponse {
  predicted_label: string
  probabilities: Record<string, number>
  model_version: string
  explanations: Record<string, number> | null
}

/** Metadatos del modelo servido (GET /model-info). */
export interface ModelInfo {
  model_loaded: boolean
  version: string | null
  message: string
  model_type: string | null
  classes: string[] | null
  test_macro_f1: number | null
}

/** Una ventana de ejemplo pre-cargada (frontend/src/examples.json). */
export interface ExampleWindow {
  id: string
  true_label: string
  tabular: TabularFeatures
  resnet_features: number[]
}
