import datetime

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from MysqlModel import MysqlModel

SEMILLA = 42
np.random.seed(SEMILLA)  # Establece la semilla aleatoria de NumPy
tf.random.set_seed(SEMILLA)  # Establece la semilla aleatoria de TensorFlow


class RNNModel:
    def __init__(self, input_file):
        self.input_file = input_file
        self.data = None
        # Guardamos el escalador para poder revertir la normalización
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.input_steps = 90
        self.output_steps = 30
        self.best_model = None
        self.n_features = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.output_column_names = None
        self.fitxerModel = '.\\model\\model.h5'
        self.hyperparameter_ranges = {
            "n_layers": (1, 3),
            "num_units_layer": (10, 300),
            "lr": (1e-5, 1e-2),
            "n_epochs": (10, 200),
            "weights": (0.1, 1.0)
        }

    def read_data(self,  nomFitxer=None):
        if nomFitxer is None:
           nomFitxer = self.input_file
        """Lee el archivo CSV"""
        self.data = pd.read_csv(nomFitxer, sep=";", parse_dates=[0])
        return self.data

    # def load_data(self):
    #     """Carga los datos"""
    #     data = None
    #     if self.read_data() is not None:
    #         data = self.read_data()
    #     return data

    def load_data(self, adressFile=None):
        """Carga los datos"""
        data = pd.read_csv(adressFile, sep=";", parse_dates=[0])
        return data

    def preprocess_data(self, data_prediccion=None):
        """
        Preprocesa los datos originales, normaliza los valores numéricos, codifica las variables categóricas y
        crea secuencias de entrada y salida.
        """
        if data_prediccion is None:
            data_procesada = self.data.copy()
        else:
            data_procesada = data_prediccion.copy()

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

        return data_procesada

    def split_data(self, data_procesada):
        """ Divide los datos en conjuntos de entrenamiento y prueba """

        # Calculamos el índice donde se dividirán los datos en conjuntos de entrenamiento y prueba
        train_size = int(len(data_procesada) * 0.7)

        train_data = data_procesada[:train_size]
        test_data = data_procesada[train_size:]

        # Crear secuencias para el conjunto de entrenamiento
        self.X_train, self.y_train = self.create_sequences(train_data)
        self.n_features = self.X_train.shape[2]
        # Crear secuencias para el conjunto de prueba
        self.X_test, self.y_test = self.create_sequences(test_data)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def create_sequences(self, data):
        """
               Crea secuencias de datos de entrada y salida a partir del conjunto de datos procesado.
        """
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

    def create_model(self, trial, n_layers, num_units_layer, lr, X_train, y_train, n_epochs, X_test, y_test,
                     batch_size):
        """
        Crea un modelo LSTM con los hiperparámetros proporcionados.
        """
        n_features = X_train.shape[2]
        model = Sequential()
        model.add(LSTM(num_units_layer, input_shape=(self.input_steps, n_features), return_sequences=True))

        for i in range(n_layers - 1):
            model.add(LSTM(num_units_layer, return_sequences=True))
        model.add(LSTM(num_units_layer, return_sequences=False))
        model.add(Dense(self.output_steps * y_train.shape[2], activation=None))
        model.add(Reshape((self.output_steps, y_train.shape[2])))

        model.compile(optimizer=Adam(learning_rate=lr), loss='mae')

        # Devuelve solo el modelo
        return model

    def objective(self, trial):
        """
        Define el objetivo que Optuna debe minimizar. Este método entrena un modelo con hiperparámetros sugeridos
        por Optuna y devuelve el error absoluto medio ponderado.
        """
        n_layers = trial.suggest_int('n_layers', 1, 3)
        num_units_layer = trial.suggest_int('num_units_layer', 30, 100)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        n_epochs = trial.suggest_int('n_epochs', 10, 100)
        batch_size = trial.suggest_int('batch_size', 16, 128)

        model = self.create_model(trial, n_layers, num_units_layer, lr, X_train, y_train, n_epochs, X_test, y_test,
                                  batch_size)

        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
        model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                  callbacks=[early_stop, pruning_callback])

        y_pred = model.predict(X_test)
        weights = np.full(y_test.shape[2], 1 / y_test.shape[2])
        weighted_mae = np.mean(
            [np.sum(weights * np.abs(y_test[:, t] - y_pred[:, t])) / np.sum(weights)
             for t in range(self.output_steps) for _ in range(y_test.shape[2])])

        # Si es el primer trial o si el modelo es mejor que el mejor encontrado hasta ahora, guarda el modelo
        if trial.number == 0 or weighted_mae < trial.study.best_value:
            self.best_model = model

        return weighted_mae

    def predict(self, model, input_sequence):
        # Realizar la predicción utilizando el modelo entrenado
        y_pred = model.predict(input_sequence)

        n_points = y_pred.shape[1] * y_pred.shape[2]
        y_pred_flat = y_pred.reshape((y_pred.shape[0], n_points))

        # Asegúrate de que el objeto scaler_output tenga la forma correcta
        if self.scaler_output.n_features_in_ != n_points:
            self.scaler_output.n_features_in_ = n_points
            self.scaler_output.min_ = np.tile(self.scaler_output.min_, self.output_steps)
            self.scaler_output.scale_ = np.tile(self.scaler_output.scale_, self.output_steps)

        y_pred_inv = self.scaler_output.inverse_transform(y_pred_flat)

        # Reorganizar las columnas de y_pred_inv de acuerdo a las características originales
        y_pred_inv_rearranged = []
        for i in range(0, y_pred_inv.shape[1], self.output_steps):
            y_pred_inv_rearranged.append(y_pred_inv[:, i:i + self.output_steps])

        y_pred_inv = np.hstack(y_pred_inv_rearranged)

        return y_pred_inv

    def predict_last_rows(self):
        """Ejecutar la predicción del mejor modelo con un fichero y seleccionar las últimas filas."""

        # Cargar los datos del fichero
        data_copiada = self.data.copy()
        calcular_tamany = self.input_steps + self.output_steps
        data_limitat = data_copiada.tail(calcular_tamany)

        data_procesada = self.preprocess_data(data_prediccion=data_limitat)

        x_prediccio, y_prediccio = self.create_sequences(data_procesada)

        n_features = x_prediccio.shape[2]

        y_pred = self.predict(self.best_model, x_prediccio)


        # Organizar y guardar la predicción en un archivo
        column_names = ['Index'] + [f'Column_{i}' for i in range(1, 4)]
        repeated_numbers = np.tile(np.arange(1, 6), len(y_pred[0]) // (3 * 5) + 1)[:len(y_pred[0]) // 3]
        y_pred_reshaped = np.column_stack((repeated_numbers, y_pred[0][::3], y_pred[0][1::3], y_pred[0][2::3]))
        output_df = pd.DataFrame(y_pred_reshaped, columns=column_names)

        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_file = f".\\predicciones\\primer_semestre{current_time}.xlsx"
        output_df.to_excel(output_file, index=False)
        print(f"Predicción guardada en el archivo: {output_file}")


if __name__ == '__main__':
    usar_optuna = True
    prediccio = True
    guardar_model = True

    rnn_model = RNNModel(input_file=".\\dades\\Dades_Per_entrenar.csv")
    rnn_model.read_data()
    if usar_optuna:
        # Leer los datos
        data_preprocessada = rnn_model.preprocess_data()
        X_train, y_train, X_test, y_test = rnn_model.split_data(data_preprocessada)

        # Ejecutar la optimización de Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(rnn_model.objective, n_trials=4)

        # Obtener y mostrar los mejores hiperparámetros encontrados por Optuna
        best_params = study.best_params
        # Crear una instancia de la clase MysqlModel
        if guardar_model:
            mySql = MysqlModel()
            mySql.guardar(rnn_model.best_model)

    if prediccio:
        # Leer los datos
        data = rnn_model.read_data(nomFitxer=".\\dades\\Dades_Per_entrenar.csv")
        # Crear una instancia de la clase MysqlModel
        mysql_model = MysqlModel()
        # Obtener los datos del fichero de predicción
        fitxerModel = mysql_model.recuperar()
        rnn_model.best_model = load_model(fitxerModel)
        # Ejecutar la predicción
        rnn_model.predict_last_rows()



