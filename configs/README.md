# configs/

Configuración del proyecto en YAML, cargada con `pydantic-settings` (sin números
mágicos en el código). Convención: las variables de entorno (con prefijo por servicio,
p. ej. `API_`) **pisan** los valores del YAML; por eso en contenedores la fuente
canónica es el entorno (12-factor) y el YAML es una comodidad para desarrollo local.

- `api.yaml` — configuración del servicio FastAPI (la consume `backend/src/config.py`).
- `dataset.yaml` — spec del dataset (partidos, ventanas, background, clips).
- `train_clips.yaml` — entrenamiento del CNN de clips (backbone, fine-tune, augmentation).
- `splits.yaml` — asignación de splits por `game_id` (generada, anti data-leakage).
