import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.dirname(__file__)

SEQUENCE_LENGTH = 50
RUL_THRESHOLD = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 15
LR = 0.001

COL_NAMES = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
DROP_SETTINGS = ['setting_1', 'setting_2', 'setting_3']
FEATURE_COLS = [c for c in COL_NAMES if c not in DROP_SENSORS + DROP_SETTINGS + ['engine_id', 'cycle']]


def load_data(filepath):
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COL_NAMES)
    return df


def compute_rul(df):
    max_cycle = df.groupby('engine_id')['cycle'].transform('max')
    df['rul'] = max_cycle - df['cycle']
    df['failure_soon'] = (df['rul'] <= RUL_THRESHOLD).astype(int)
    return df


def scale_features(train_df, val_df, feature_cols):
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    return train_scaled, val_scaled, scaler


def create_sequences(df, feature_cols, sequence_length):
    sequences = []
    labels = []
    engine_ids = []
    for engine_id, group in df.groupby('engine_id'):
        group = group.sort_values('cycle')
        data = group[feature_cols].values
        target = group['failure_soon'].values
        if len(data) < sequence_length:
            continue
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            labels.append(target[i + sequence_length - 1])
            engine_ids.append(engine_id)
    return np.array(sequences), np.array(labels), np.array(engine_ids)


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall'])
    return model


def find_optimal_threshold(y_true, y_pred, target_recall=0.95):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    idx = np.where(recall >= target_recall)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(precision[idx])]
        return float(thresholds[best_idx])
    return 0.5


def main():
    print("Loading training data...")
    train_df = load_data(os.path.join(DATA_DIR, 'train_FD001.txt'))
    train_df = compute_rul(train_df)

    print("Splitting engines 80/20...")
    engine_ids = train_df['engine_id'].unique()
    train_engines, val_engines = train_test_split(engine_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df_split = train_df[train_df['engine_id'].isin(train_engines)]
    val_df_split = train_df[train_df['engine_id'].isin(val_engines)]

    print("Scaling features...")
    train_scaled, val_scaled, scaler = scale_features(train_df_split, val_df_split, FEATURE_COLS)

    print("Creating sequences...")
    X_train, y_train, _ = create_sequences(train_scaled, FEATURE_COLS, SEQUENCE_LENGTH)
    X_val, y_val, _ = create_sequences(val_scaled, FEATURE_COLS, SEQUENCE_LENGTH)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    print("Building model...")
    model = build_model((SEQUENCE_LENGTH, len(FEATURE_COLS)))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_recall', mode='max', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_recall', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'model.keras'), monitor='val_recall', mode='max', save_best_only=True, verbose=1)
    ]

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("Finding optimal threshold for recall >= 0.95...")
    val_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_val, val_pred, target_recall=0.95)
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    print("Saving artifacts...")
    joblib.dump({
        'scaler': scaler,
        'feature_names': FEATURE_COLS,
        'feature_indices': {name: i for i, name in enumerate(FEATURE_COLS)}
    }, os.path.join(MODEL_DIR, 'preprocessing.pkl'))

    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'threshold': optimal_threshold,
        'n_features': len(FEATURE_COLS),
        'feature_names': FEATURE_COLS,
        'risk_levels': {
            'stable': [0.0, 0.3],
            'moderate': [0.3, 0.7],
            'critical': [0.7, 1.0]
        },
        'rul_threshold': RUL_THRESHOLD,
        'target_recall': 0.95
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()