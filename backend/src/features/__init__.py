"""Feature engineering — the SINGLE source of truth for preprocessing (invariant 3).

All feature/transform logic lives here and is imported by BOTH training and the API.
Fitted transformers (scalers, encoders, image-transform config) are serialized with the
model and loaded at serving time — never re-fitted in the API. This is what prevents
training-serving skew.

Tabular point-in-time features (invariant 2) arrive in Fase 1; image transforms in
Fase 2/3. This module is intentionally empty in Fase 0.
"""
