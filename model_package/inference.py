import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


class FailurePredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(__file__)
        self.model_dir = model_dir

        self.model = load_model(os.path.join(model_dir, 'model.keras'))
        preprocessing = joblib.load(os.path.join(model_dir, 'preprocessing.pkl'))
        self.scaler = preprocessing['scaler']
        self.feature_names = preprocessing['feature_names']
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.sequence_length = self.metadata['sequence_length']
        self.threshold = self.metadata['threshold']
        self.n_features = self.metadata['n_features']

    def _to_df(self, array):
        if isinstance(array, np.ndarray):
            return pd.DataFrame(array, columns=self.feature_names)
        return array

    def _preprocess(self, sequence):
        if sequence.shape != (self.sequence_length, self.n_features):
            raise ValueError(f"Expected shape ({self.sequence_length}, {self.n_features}), got {sequence.shape}")
        df = self._to_df(sequence)
        scaled = self.scaler.transform(df)
        return scaled.reshape(1, self.sequence_length, self.n_features)

    def predict(self, sequence: np.ndarray) -> float:
        x = self._preprocess(sequence)
        risk = float(self.model.predict(x, verbose=0)[0, 0])
        return risk

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3 or sequences.shape[1:] != (self.sequence_length, self.n_features):
            raise ValueError(f"Expected shape (batch, {self.sequence_length}, {self.n_features}), got {sequences.shape}")
        scaled = np.array([self.scaler.transform(self._to_df(s)) for s in sequences])
        risks = self.model.predict(scaled, verbose=0).flatten()
        return risks

    def classify(self, risk: float) -> str:
        if risk < 0.3:
            return 'stable'
        elif risk < 0.7:
            return 'moderate'
        return 'critical'

    def estimate_rul(self, risk: float) -> int:
        if risk > 0.8:
            return 10
        elif risk > 0.5:
            return 30
        elif risk > 0.3:
            return 60
        return 100

    def get_metadata(self):
        return self.metadata.copy()