# Phase 5 — Explainability Layer (XAI)

## Purpose
SOC analysts must know *why* an event was flagged. A score alone is unactionable. This module provides per-alert explanations combining machine learning attributions with natural language narratives.

## Techniques
- **SHAP (TreeExplainer):** Local feature attributions for the Phase 4 XGBoost multi-class threat classifier, cached on disk at `models/shap_explainer.pkl`.
- **Sequence Reconstruction Error Breakdown:** Per-dimension error attribution from the Phase 3 sequence autoencoder.
- **Deterministic Rule Hit Tags:** Prominently injects rule-hit tags (`impossible_travel`, `brute_force`) when deterministic overrides trigger.
- **Natural Language Translator:** Converts raw feature names and values into plain English analyst narratives.

## Output Contract (`explain(event_id)`)
```json
{
  "event_id": "8f3b2a1c-...",
  "risk_score": 0.87,
  "attack_type": "impossible_travel",
  "reasons": [
    {
      "feature": "geo_velocity_kmh",
      "value": 2400.0,
      "narrative": "Travel velocity of 2400.0 km/h between consecutive logins exceeds physical speed limits (>900 km/h)"
    },
    {
      "feature": "geo_distance_prev_km",
      "value": 2100.5,
      "narrative": "Physical distance of 2100.5 km from previous login location exceeds physical movement threshold (>500 km)"
    },
    {
      "feature": "device_fingerprint_hash_novelty",
      "value": 1.0,
      "narrative": "Access from an unrecognized device fingerprint or operating system"
    }
  ],
  "entity_history_snippet": [
    {
      "event_id": "...",
      "timestamp": "2026-05-10T14:20:00.000000Z",
      "resource_accessed": "/api/v1/auth/login",
      "source_ip": "192.168.1.10",
      "auth_method": "password",
      "session_duration": 180
    }
  ],
  "cold_start": false
}
```

## CLI Usage
To explain a specific event by ID:
```bash
python -m src.phase5_explainability.explain --event-id <EVENT_ID>
```

To run the automated unit test suite across 3 seeded events of distinct attack types:
```bash
python -m src.phase5_explainability.explain
```

## Known Limitation
SHAP values on sequence-model outputs are approximated via per-timestep reconstruction error breakdown, not gradient-based integrated gradients.
