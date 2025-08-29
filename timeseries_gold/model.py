from __future__ import annotations
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense


class ModelBuilder:
    @staticmethod
    def lstm_stacked(window_size: int, n_features: int, n_targets: int) -> Model:
        inp = Input(shape=(window_size, n_features))
        x = LSTM(128, return_sequences=True)(inp)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(n_targets)(x)
        model = Model(inp, out)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
