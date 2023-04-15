import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from MysqlModel import MysqlModel
import tensorflow.keras.backend as K
from optuna.pruners import MedianPruner
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu

import tensorflow as tf

def clipped_relu(x):
    return tf.keras.activations.relu(x, max_value=0)






from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

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
        self.best_trial_value = None
        self.n_features = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.output_column_names = None
        self.fitxerModel = '.\\model\\model.h5'
        tf.keras.utils.get_custom_objects().update({'clipped_relu': clipped_relu})
        self.hyperparameter_ranges = {
            "n_layers": (1, 5),
            "num_units_layer": (10, 200),
            "lr": (1e-5, 1e-2),
            "n_epochs": (10, 200),
            "layer_weights": (0.1, 1.0),
            "batch_size": (16, 128)
        }

    def read_data(self, nomFitxer=None):
        if nomFitxer is None:
            nomFitxer = self.input_file
        """Lee el archivo CSV"""
        self.data = pd.read_csv(nomFitxer, sep=";", parse_dates=[0])
        return self.data

    def load_data(self, adressFile=None):
        """Carga los datos"""
        data = pd.read_csv(adressFile, sep=";", parse_dates=[0])
        return data

    def preprocess_data(self, data_prediccion=None):
        if data_prediccion is None:
            data_procesada = self.data.copy()
        else:
            data_procesada = data_prediccion.copy()

        data_procesada[data_procesada.columns[2:]] = data_procesada.iloc[:, 2:].astype(float)
        data_procesada = pd.get_dummies(data_procesada, columns=['Gran Grup'], prefix='Gran Grup')

        # Crear una lista de escaladores MinMax para cada categoría
        self.scalers_input = [MinMaxScaler() for _ in range(5)]

        for i in range(5):
            # Aplicar el escalador correspondiente a las filas de cada categoría
            category_mask = (data_procesada[f'Gran Grup_{i + 1}'] == 1)
            data_procesada.loc[category_mask, data_procesada.columns[4:-5]] = self.scalers_input[i].fit_transform(
                data_procesada.loc[category_mask, data_procesada.columns[4:-5]])

        data_procesada.iloc[:, 1:4] = self.scaler_output.fit_transform(data_procesada.iloc[:, 1:4])
        data_procesada = data_procesada.drop(data_procesada.columns[0], axis=1)
        data_procesada = data_procesada.dropna()

        return data_procesada

    def split_data(self, data_procesada):
        """ Divide los datos en conjuntos de entrenamiento y prueba """

        # Calculamos el índice donde se dividirán los datos en conjuntos de entrenamiento y prueba
        train_size = int(len(data_procesada) * 0.7)

        train_data = data_procesada[:train_size]
        test_data = data_procesada[train_size:]

        # Crear secuencias para el conjunto de entrenamiento
        self.x_train, self.y_train = self.create_sequences(train_data)
        self.n_features = self.x_train.shape[2]
        # Crear secuencias para el conjunto de prueba
        self.x_test, self.y_test = self.create_sequences(test_data)

        return self.x_train, self.y_train, self.x_test, self.y_test

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

    def create_model(self, n_layers, num_units_layer, lr, layer_weights):
        """
        Crea un modelo LSTM con los hiperparámetros proporcionados.
        """
        n_features = self.x_train.shape[2]
        model = Sequential()
        model.add(
            Bidirectional(LSTM(num_units_layer, input_shape=(self.input_steps, n_features), return_sequences=True)))

        for i in range(n_layers - 1):
            model.add(Bidirectional(LSTM(num_units_layer, return_sequences=True)))
            model.layers[-1].set_weights(
                [w * layer_weights for w in model.layers[-1].get_weights()])  # Añadir esta línea

        model.add(Bidirectional(LSTM(num_units_layer, return_sequences=False)))
        model.add(Dense(self.output_steps * self.y_train.shape[2], activation=None))
        model.add(Reshape((self.output_steps, self.y_train.shape[2])))

        # Compilar el modelo con el optimizador Adam y la función de pérdida mean_absolute_error
        model.compile(optimizer=Adam(learning_rate=lr), loss=tf.keras.losses.MeanAbsoluteError())

        # Devuelve solo el modelo
        return model

    def objective(self, trial):
        """
        Define el objetivo que Optuna debe minimizar. Este método entrena un modelo con hiperparámetros sugeridos
        por Optuna y devuelve el error absoluto medio ponderado.
        """
        n_layers = trial.suggest_int("n_layers", self.hyperparameter_ranges["n_layers"][0],
                                     self.hyperparameter_ranges["n_layers"][1])
        num_units_layer = trial.suggest_int("num_units_layer", self.hyperparameter_ranges["num_units_layer"][0],
                                            self.hyperparameter_ranges["num_units_layer"][1])
        lr = trial.suggest_float("lr", self.hyperparameter_ranges["lr"][0], self.hyperparameter_ranges["lr"][1],
                                 log=True)
        n_epochs = trial.suggest_int("n_epochs", self.hyperparameter_ranges["n_epochs"][0],
                                     self.hyperparameter_ranges["n_epochs"][1])
        batch_size = trial.suggest_int("batch_size", self.hyperparameter_ranges["batch_size"][0],
                                       self.hyperparameter_ranges["batch_size"][1])

        layer_weights = trial.suggest_float("layer_weights", self.hyperparameter_ranges["layer_weights"][0],
                                            self.hyperparameter_ranges["layer_weights"][1])

        model = self.create_model(n_layers, num_units_layer, lr, layer_weights)

        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

        history = model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=batch_size, verbose=0,
                            validation_data=(self.x_test, self.y_test), callbacks=[early_stop, pruning_callback])

        # Evaluar el modelo en el conjunto de prueba
        y_pred = model.predict(self.x_test)
        # Calcular el error absoluto medio ponderado
        weighted_mae = self.weighted_mean_absolute_error(self.y_test, y_pred)

        # Comprobar si este modelo es el mejor hasta ahora y, de ser así, guardarlo
        if self.best_trial_value is None or weighted_mae < self.best_trial_value:
            self.best_trial_value = weighted_mae
            self.best_model = model

        return weighted_mae

    def weighted_mean_absolute_error(self, y_true, y_pred):
        """
        Calcula el error absoluto medio ponderado, dando más peso a los errores
        en las primeras etapas de la secuencia de salida.
        """
        absolute_errors = np.abs(y_true - y_pred)
        weights = np.array([1 / (i + 1) for i in range(self.output_steps)]).reshape(-1, 1)
        weighted_absolute_errors = absolute_errors * weights
        return np.mean(weighted_absolute_errors)

    def optimize(self, n_trials=50):
        """
        Optimiza los hiperparámetros del modelo LSTM utilizando Optuna.
        """
        pruner = MedianPruner()
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params

        return study.best_params

    def predict(self, model, input_sequence):
        # Realizar la predicción utilizando el mejor modelo encontrado

        if model is None:
            model = self.best_model

        data_procesada = self.preprocess_data(input_sequence)

        x, y = self.create_sequences(data_procesada)

        y_pred = model.predict(x)

        n_points = y_pred.shape[1] * y_pred.shape[2]
        y_pred_flat = y_pred.reshape((y_pred.shape[0], n_points))

        y_pred_inv = np.zeros(y_pred_flat.shape)

        for i in range(5):
            # Aplicar el escalador inverso correspondiente a las filas de cada categoría
            category_mask = (input_sequence[:, -1, -5:] == [1 if j == i else 0 for j in range(5)]).all(axis=1)
            y_pred_inv[category_mask] = self.scalers_input[i].inverse_transform(y_pred_flat[category_mask])

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


        y_pred = self.predict(self.best_model, data_limitat)


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
    input_file = ".\\dades\\Dades_Per_entrenar.csv"  # Reemplazar por la ruta del archivo CSV
    rnn = RNNModel(input_file)
    data = rnn.read_data()
    if usar_optuna:
        data_procesada = rnn.preprocess_data()
        rnn.split_data(data_procesada)

        print("Optimizando hiperparámetros...")
        best_params = rnn.optimize(n_trials=50)
        print(f"Mejores hiperparámetros encontrados: {best_params}")
        print(best_params)

        final_model = rnn.best_model

        if guardar_model:
            print("Guardando el modelo...")
            final_model.save(rnn.fitxerModel)
            print(f"Modelo guardado en {rnn.fitxerModel}")
            # mySql = MysqlModel()
            # mySql.guardar(final_model)

    if prediccio:
        # Leer los datos
        print("Leyendo los datos...")
        data = rnn.read_data(nomFitxer=".\\dades\\Dades_Per_entrenar.csv")
        # Crear una instancia de la clase MysqlModel
        # mysql_model = MysqlModel()
        # Obtener los datos del fichero de predicción
        # fitxerModel = mysql_model.recuperar()
        rnn.best_model = load_model(rnn.fitxerModel)
        # Ejecutar la predicción
        print("Ejecutando la predicción...")
        rnn.predict_last_rows()
