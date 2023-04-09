import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

SEMILLA = 42
np.random.seed(SEMILLA)  # Establece la semilla aleatoria de NumPy
tf.random.set_seed(SEMILLA)  # Establece la semilla aleatoria de TensorFlow

class RNNModel:
    def __init__(self, input_file):
        self.input_file = input_file
        # Guardamos el escalador para poder revertir la normalización
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.input_steps = 36
        self.output_steps = 12

    def read_data(self):
        """Lee el archivo CSV"""
        data = pd.read_csv(self.input_file, sep=";", parse_dates=[0])
        return data

    def preprocess_data(self, data):
        """Define la columna de fecha, categórica y numérica"""
        data_procesada = data.copy()

        # las columnas numéricas a float
        data_procesada[data_procesada.columns[2:]] = data_procesada.iloc[:, 2:].astype(float)

        # Codificar columna categórica 'Gran Grup' usando One-Hot Encoding
        data_procesada = pd.get_dummies(data_procesada, columns=['Gran Grup'], prefix='Gran Grup')

        # Normalizar los datos numéricos de entrada
        data_procesada.iloc[:, 4:-5] = self.scaler_input.fit_transform(data_procesada.iloc[:, 4:-5])

        # Normalizar las columnas de salida numéricas (2ª , 3, 4 columna)
        data_procesada.iloc[:, 1:4] = self.scaler_output.fit_transform(data_procesada.iloc[:, 1:4])

        # Eliminar la columna de fecha
        data_procesada = data_procesada.drop(data_procesada.columns[0], axis=1)

        # Elimina los valores NaN
        data_procesada = data_procesada.dropna()

        # Calculamos el índice donde se dividirán los datos en conjuntos de entrenamiento y prueba
        train_size = int(len(data_procesada) * 0.8)

        train_data = data_procesada[:train_size]
        test_data = data_procesada[train_size:]

        self.input_steps = 12
        self.output_steps = 1

        # Crear secuencias para el conjunto de entrenamiento
        X_train, y_train = self.create_sequences(train_data)

        # Crear secuencias para el conjunto de prueba
        X_test, y_test = self.create_sequences(test_data)

        return X_train, y_train, X_test, y_test

    def create_sequences(self, data):
        """Crea secuencias de datos de entrada y salida"""
        X, y = [], []

        # Recorrer el conjunto de datos y crear secuencias de entrada y salida
        for i in range(len(data) - self.input_steps - self.output_steps + 1):
            # Seleccionar las columnas de entrada (todas menos las tres numéricas que se quieren predecir)
            input_data = data.iloc[i:i + self.input_steps, [0, *range(3, data.shape[1])]].values
            # Seleccionar las columnas de salida (numéricas)
            output_data = data.iloc[i + self.input_steps:i + self.input_steps + self.output_steps, 0:3].values
            self.output_column_names = data.columns[3:6].tolist()

            X.append(input_data)
            y.append(output_data)

        return np.array(X), np.array(y)

    def create_model(self, n_layers, num_units_layer, lr, n_features, X_train, y_train, X_test, y_test):
        # Definir y entrenar el modelo LSTM
        model = Sequential()

        for i in range(n_layers - 1):
            model.add(LSTM(num_units_layer, input_shape=(self.input_steps, n_features), return_sequences=True))
        model.add(LSTM(num_units_layer, return_sequences=False))
        model.add(Dense(self.output_steps * y_train.shape[2], activation=None))
        model.add(Reshape((self.output_steps, y_train.shape[2])))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
        y_pred = model.predict(X_test)
        weights = np.full(y_test.shape[2], 1 / y_test.shape[2])
        weighted_mae = np.mean(
            [np.sum(weights * np.abs(y_test[:, t] - y_pred[:, t])) / np.sum(weights)
             for t in range(self.output_steps) for _ in range(y_test.shape[2])])
        print(f"El error absoluto medio ponderado (weighted MAE) en el conjunto de prueba es: {weighted_mae}")

        return model, y_pred

    # Guardar las predicciones en un archivo xlsx
    def save_predictions(self, y_test, y_pred, filename):
        # Aplanar y_test y y_pred a dos dimensiones
        y_test_flat = y_test.reshape(y_test.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

        # Revertir la normalización para y_test_flat
        y_test_inv = []
        for i in range(0, y_test_flat.shape[1], 3):
            temp = self.scaler_output.inverse_transform(y_test_flat[:, i:i + 3])
            y_test_inv.append(temp)
        y_test_inv = np.hstack(y_test_inv)

        # Revertir la normalización para y_pred_flat
        y_pred_inv = []
        for i in range(0, y_pred_flat.shape[1], 3):
            temp = self.scaler_output.inverse_transform(y_pred_flat[:, i:i + 3])
            y_pred_inv.append(temp)
        # Aplanar y_pred_inv
        y_pred_inv = np.hstack(y_pred_inv)

        # Crear un DataFrame de pandas con la entrada aplanada
        data = np.empty((y_test_inv.shape[0], y_test_inv.shape[1] * 2), dtype=y_test_inv.dtype)

        n_variables = 3
        n_points = y_test_inv.shape[1] // n_variables

        for i in range(n_points):
            for j in range(n_variables):
                data[:, i * n_variables * 2 + j] = y_test_inv[:, i * n_variables + j]
                data[:, i * n_variables * 2 + j + n_variables] = y_pred_inv[:, i * n_variables + j]

        columns = [f"{self.output_column_names[i]}_{t}_{i + 1}_{j + 1}" for i in range(n_points) for t in ['test', 'pred']
                   for j in
                   range(n_variables)]

        results = pd.DataFrame(data, columns=columns)

        # Guardar el DataFrame en un archivo Excel
        results.to_excel(filename, index=False)

    # Graficar las predicciones y los valores reales de cada variable
    def plot_predictions(self, y_test, y_pred):
        """Grafica las predicciones y los valores reales de cada variable"""

        variables = ['Variable_1', 'Variable_2', 'Variable_3']

        for i in range(y_test.shape[2]):
            plt.figure(figsize=(12, 6))
            plt.plot(y_test[:, :, i], label=f"{variables[i]} real")
            plt.plot(y_pred[:, :, i], label=f"{variables[i]} predicho", linestyle="--")
            plt.legend()
            plt.xlabel("Temps")
            plt.ylabel("Valor")
            plt.title(f"Comparació de dades reals i prediccions per a {self.output_column_names[i]}")
            current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
            plt.savefig(f".\\GRAFIC\\grafic_{self.output_column_names[i]}_{current_time}.png")


if __name__ == '__main__':
    rnn_model = RNNModel(input_file=".\\dades\\Dades_Per_entrenar.csv")
    data = rnn_model.read_data()
    X_train, y_train, X_test, y_test = rnn_model.preprocess_data(data)

    # Establecer los hiperparámetros del modelo
    n_layers = 2
    num_units_layer = 64
    lr = 0.001
    n_features = X_train.shape[2]

    model, y_pred = rnn_model.create_model(n_layers, num_units_layer, lr, n_features, X_train, y_train,
                           X_test, y_test)

    # Guardar las predicciones en un archivo CSV
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    rnn_model.save_predictions(y_test, y_pred, (f".\\previsio\\prediccions_RNNModel_2023_{current_time}.xlsx"))

    # Graficar las predicciones y los valores reales
    rnn_model.plot_predictions(y_test, y_pred)
