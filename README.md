# Aircraft Engine Failure Prediction Model(Deep learning project)

Predicts probability of engine failure within 30 flight cycles using LSTM on NASA CMAPSS FD001 sensor data.

## Quick Start

```bash
pip install -r requirements.txt

# Train model (run once)
python -m model_package.train

# Use in your backend
from model_package import FailurePredictor

predictor = FailurePredictor()
risk = predictor.predict(sensor_sequence)  # (50, 14) -> float 0.0-1.0
level = predictor.classify(risk)            # 'stable' | 'moderate' | 'critical'
rul = predictor.estimate_rul(risk)          # estimated cycles remaining
```

## Input Format

`predict()` expects a **numpy array of shape (50, 14)** — the last 50 cycles of 14 sensor readings (already filtered from raw 21 sensors).

Feature order (matching training):
```
sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9,
sensor_11, sensor_12, sensor_13, sensor_14, sensor_15,
sensor_17, sensor_20, sensor_21
```

## Output

| Method | Returns |
|--------|---------|
| `predict(seq)` | Risk score 0.0–1.0 (1.0 = failure imminent) |
| `predict_batch(seqs)` | Array of risk scores for batch |
| `classify(risk)` | `'stable'` (<0.3), `'moderate'` (0.3–0.7), `'critical'` (≥0.7) |
| `estimate_rul(risk)` | Heuristic cycles remaining: >0.8→10, >0.5→30, >0.3→60, else 100+ |

## Model Details

- **Architecture**: LSTM(100) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1, sigmoid)
- **Training**: 80/20 engine split, sequence length 50, binary label (RUL ≤ 30)
- **Threshold**: Optimized for recall ≥ 0.95 on validation set
- **Format**: Saved as `model_package/model.keras` (Keras native)

## Files

```
model_package/
├── train.py              # Run once to train
├── inference.py          # Import this: FailurePredictor
├── preprocessing.pkl     # Fitted MinMaxScaler + feature info
├── model.keras           # Trained model
├── metadata.json         # Thresholds, feature names, config
└── __init__.py
```
