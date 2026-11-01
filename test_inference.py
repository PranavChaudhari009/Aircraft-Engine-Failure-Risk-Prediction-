import numpy as np
import pandas as pd
from model_package import FailurePredictor
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

COL_NAMES = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
DROP_SETTINGS = ['setting_1', 'setting_2', 'setting_3']
FEATURE_COLS = [c for c in COL_NAMES if c not in DROP_SENSORS + DROP_SETTINGS + ['engine_id', 'cycle']]

test_df = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=COL_NAMES)
predictor = FailurePredictor()

engine3 = test_df[test_df['engine_id'] == 3].sort_values('cycle')
seq_early = engine3[FEATURE_COLS].iloc[:50].values.astype(np.float32)
seq_late = engine3[FEATURE_COLS].iloc[-50:].values.astype(np.float32)

risk_early = predictor.predict(seq_early)
risk_late = predictor.predict(seq_late)

print(f'Engine 3 (first 50): risk={risk_early:.4f}, level={predictor.classify(risk_early)}, rul={predictor.estimate_rul(risk_early)}')
print(f'Engine 3 (last 50): risk={risk_late:.4f}, level={predictor.classify(risk_late)}, rul={predictor.estimate_rul(risk_late)}')
sys.stdout.flush()