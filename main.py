import datetime
import os
import pickle
import shutil

import h5py
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam

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
        self.input_steps = 18
        self.output_steps = 6
        self.input_steps_categoria = 90
        self.output_steps_categoria = 30
        self.best_model = None
        self.best_trial_value = None
        self.n_features = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.output_column_names = None
        self.fitxerModel = '.\\model\\model.h5'
        self.hyperparameter_ranges = {
            "n_layers": [1, 30],
            "num_units_layer": [16, 200],
            "lr": [1e-4, 1e-2],
            "n_epochs": [100, 300],
            "batch_size": [16, 64]
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

    def preprocess_data_sin_categoria(self, data_prediccion=None):
        if data_prediccion is None:
            data_procesada = self.data.copy()
        else:
            data_procesada = data_prediccion.copy()

        data_procesada[data_procesada.columns[1:]] = data_procesada.iloc[:, 1:].astype(float)
        data_procesada.iloc[:, 4:] = self.scaler_input.fit_transform(data_procesada.iloc[:, 4:])
        data_procesada.iloc[:, 1:4] = self.scaler_output.fit_transform(data_procesada.iloc[:, 1:4])
        data_procesada = data_procesada.drop(data_procesada.columns[0], axis=1)
        data_procesada = data_procesada.dropna()

        return data_procesada

    def preprocess_data(self, data_prediccion=None):
        if data_prediccion is None:
            data_procesada = self.data.copy()
        else:
            data_procesada = data_prediccion.copy()

        data_procesada[data_procesada.columns[2:]] = data_procesada.iloc[:, 2:].astype(float)
        data_procesada = pd.get_dummies(data_procesada, columns=['Gran Grup'], prefix='Gran Grup')
        data_procesada.iloc[:, 4:-5] = self.scaler_input.fit_transform(data_procesada.iloc[:, 4:-5])
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

    def create_sequences_for_prediction(self, data):
        """
        Crea secuencias de datos de entrada a partir del conjunto de datos procesado para realizar predicciones.
        """
        # Asegurarse de que los datos contienen al menos los últimos 'input_steps_categoria'
        if len(data) < self.input_steps_categoria:
            raise ValueError(
                f"Se requieren al menos {self.input_steps_categoria} registros en los datos para la predicción.")

        # Tomar solo los últimos 'input_steps_categoria' datos
        input_data = data.iloc[-self.input_steps_categoria:, [0, *range(4, data.shape[1])]].values

        return np.array([input_data])

    def create_sequences_sin_categoria(self, data):
        """
               Crea secuencias de datos de entrada y salida a partir del conjunto de datos procesado.
        """
        X, y = [], []

        # Recorrer el conjunto de datos y crear secuencias de entrada y salida
        for i in range(len(data) - self.input_steps - self.output_steps + 1):
            # Seleccionar las columnas de entrada (todas menos las tres numéricas que se quieren predecir)
            input_data = data.iloc[i:i + self.input_steps, [0, *range(4, data.shape[1])]].values
            # Seleccionar las columnas de salida (numéricas)
            output_data = data.iloc[i + self.input_steps:i + self.input_steps + self.output_steps, 0:3].values
            self.output_column_names = data.columns[3:6].tolist()

            X.append(input_data)
            y.append(output_data)

        return np.array(X), np.array(y)

    def create_sequences(self, data):
        """
               Crea secuencias de datos de entrada y salida a partir del conjunto de datos procesado.
        """
        X, y = [], []

        # Recorrer el conjunto de datos y crear secuencias de entrada y salida
        for i in range(len(data) - self.input_steps_categoria - self.output_steps_categoria + 1):
            # Seleccionar las columnas de entrada (todas menos las tres numéricas que se quieren predecir)
            input_data = data.iloc[i:i + self.input_steps_categoria, [0, *range(4, data.shape[1])]].values
            # Seleccionar las columnas de salida (numéricas)
            output_data = data.iloc[
                          i + self.input_steps_categoria:i + self.input_steps_categoria + self.output_steps_categoria,
                          0:3].values
            self.output_column_names = data.columns[3:6].tolist()

            X.append(input_data)
            y.append(output_data)

        return np.array(X), np.array(y)

    def create_model(self, n_layers, num_units_layer, lr):
        """
        Crea un modelo LSTM con los hiperparámetros proporcionados.
        """
        n_features = self.x_train.shape[2]
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(num_units_layer, input_shape=(self.input_steps_categoria, n_features), return_sequences=True)))
        model.add(Dropout(0.2))

        for i in range(n_layers - 1):
            model.add(Bidirectional(LSTM(num_units_layer, return_sequences=True)))
            model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(num_units_layer, return_sequences=False)))
        model.add(Dense(self.output_steps_categoria * self.y_train.shape[2], activation='relu'))
        model.add(Reshape((self.output_steps_categoria, self.y_train.shape[2])))

        # Compilar el modelo con el optimizador Adam y la función de pérdida mean_absolute_error
        model.compile(optimizer=Adam(learning_rate=lr), loss=tf.keras.losses.MeanAbsoluteError())

        # Devuelve solo el modelo
        return model

    def create_model_sin_Categoria(self, n_layers, num_units_layer, lr):
        """
        Crea un modelo LSTM con los hiperparámetros proporcionados.
        """
        n_features = self.x_train.shape[2]
        model = Sequential()
        model.add(
            Bidirectional(LSTM(num_units_layer, input_shape=(self.input_steps, n_features), return_sequences=True)))
        model.add(Dropout(0.2))

        for i in range(n_layers - 1):
            model.add(Bidirectional(LSTM(num_units_layer, return_sequences=True)))
            model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(num_units_layer, return_sequences=False)))
        model.add(Dense(self.output_steps * self.y_train.shape[2], activation='relu'))
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

        model = self.create_model(n_layers, num_units_layer, lr)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=batch_size, verbose=0,
                            validation_data=(self.x_test, self.y_test), callbacks=[early_stopping])

        # Evaluar el modelo en el conjunto de prueba
        y_pred = model.predict(self.x_test)
        # Calcular el error absoluto medio ponderado
        weighted_mae = self.weighted_mean_absolute_error(self.y_test, y_pred)

        # Comprobar si este modelo es el mejor hasta ahora y, de ser así, guardarlo
        if self.best_trial_value is None or weighted_mae < self.best_trial_value:
            self.best_trial_value = weighted_mae
            self.best_model = model

        return weighted_mae

    def get_error_weights(self, output_steps):
        weights = [1 / (i + 1) for i in range(output_steps)]
        return tf.constant(weights, dtype=tf.float64)

    @tf.function
    def weighted_mean_absolute_error(self, y_true, y_pred):
        """
        Calcula el error absoluto medio ponderado, dando más peso a los errores
        en las primeras etapas de la secuencia de salida.
        """
        y_pred = tf.cast(y_pred, tf.float64)  # Convertir y_pred a float64
        y_true = tf.cast(y_true, tf.float64)  # Convertir y_true a float64
        absolute_errors = tf.math.abs(y_true - y_pred)

        weights = self.get_error_weights(self.output_steps_categoria)
        weights = tf.reshape(weights, (-1, 1))  # Asegurar que weights tenga la forma (-1, 1)
        weighted_absolute_errors = absolute_errors * weights

        return tf.reduce_mean(weighted_absolute_errors)

    @tf.function
    def weighted_mean_absolute_error_Sin_Categoria(self, y_true, y_pred):
        """
        Calcula el error absoluto medio ponderado, dando más peso a los errores
        en las primeras etapas de la secuencia de salida.
        """
        y_pred = tf.cast(y_pred, tf.float64)  # Convertir y_pred a float64
        y_true = tf.cast(y_true, tf.float64)  # Convertir y_true a float64
        absolute_errors = tf.math.abs(y_true - y_pred)

        weights = self.get_error_weights(self.output_steps)
        weights = tf.reshape(weights, (-1, 1))  # Asegurar que weights tenga la forma (-1, 1)
        weighted_absolute_errors = absolute_errors * weights

        return tf.reduce_mean(weighted_absolute_errors)

    def optimize(self, n_trials=50, study=None, initial_params=None):
        """
        Optimiza los hiperparámetros del modelo LSTM utilizando Optuna.
        """
        pruner = MedianPruner()

        if study is None:
            study = optuna.create_study(direction='minimize', pruner=pruner)

        if initial_params is not None:
            # Utilizar los valores iniciales como sugerencias en el espacio de búsqueda de Optuna
            study.enqueue_trial(initial_params)

        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params

        return study.best_params, study

    def train_best_model(self, best_params):
        """
        Entrena el modelo con los mejores hiperparámetros encontrados y devuelve el modelo entrenado.
        """
        n_layers = best_params["n_layers"]
        num_units_layer = best_params["num_units_layer"]
        lr = best_params["lr"]
        n_epochs = 10000
        batch_size = best_params["batch_size"]

        self.best_model = self.create_model(n_layers, num_units_layer, lr)

        print("Hiperparámetros del modelo entrenado:")
        for layer in self.best_model.layers:
            print(layer.get_config())

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        history = self.best_model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
                                      validation_data=(self.x_test, self.y_test))
        print(history)
        # Evaluar el modelo en el conjunto de prueba
        y_pred = self.best_model.predict(self.x_test)
        # Calcular el error absoluto medio ponderado
        weighted_mae = self.weighted_mean_absolute_error(self.y_test, y_pred)
        print(weighted_mae)

        return self.best_model

    def predict(self, model, input_sequence):
        # Realizar la predicción utilizando el modelo entrenado
        y_pred = model.predict(input_sequence)

        n_points = y_pred.shape[1] * y_pred.shape[2]
        y_pred_flat = y_pred.reshape((y_pred.shape[0], n_points))

        # Asegúrate de que el objeto scaler_output tenga la forma correcta
        if self.scaler_output.n_features_in_ != n_points:
            self.scaler_output.n_features_in_ = n_points
            self.scaler_output.min_ = np.tile(self.scaler_output.min_, self.output_steps_categoria)
            self.scaler_output.scale_ = np.tile(self.scaler_output.scale_, self.output_steps_categoria)

        y_pred_inv = self.scaler_output.inverse_transform(y_pred_flat)

        # Reorganizar las columnas de y_pred_inv de acuerdo a las características originales
        y_pred_inv_rearranged = []
        for i in range(0, y_pred_inv.shape[1], self.output_steps_categoria):
            y_pred_inv_rearranged.append(y_pred_inv[:, i:i + self.output_steps_categoria])

        y_pred_inv = np.hstack(y_pred_inv_rearranged)

        return y_pred_inv

    def predict_sin_Categoria(self, model, input_sequence):
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
        data_copiada = data_copiada.dropna()

        data_limitat = data_copiada.tail(self.input_steps_categoria)

        data_procesada = self.preprocess_data(data_prediccion=data_limitat)

        x_prediccio = self.create_sequences_for_prediction(data_procesada)

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

    def predict_last_rows_sin_Categoria(self):
        """Ejecutar la predicción del mejor modelo con un fichero y seleccionar las últimas filas."""

        # Cargar los datos del fichero
        data_copiada = self.data.copy()
        data_copiada = data_copiada.dropna()

        data_limitat = data_copiada.tail(self.input_steps_categoria)

        data_procesada = self.preprocess_data(data_prediccion=data_limitat)

        x_prediccio = self.create_sequences_for_prediction(data_procesada)

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


    def guardar_model(self, model_save, study, csv_path):
        # Obtenir la data i hora actual amb el format especificat
        data_hora = datetime.datetime.now().strftime('%d%m%Y__%H_%M')

        # # nameModel = study.best_params["nameModel"]
        # # Crear la ruta de la carpeta on es guardaran els resultats
        # carpetaModel = os.path.join(os.path.join(".\\model",  csv_path))
        # os.makedirs(carpetaModel, exist_ok=True)

        # Crear la subcarpeta amb el nom de la data i hora
        subcarpeta = os.path.join(os.path.join(".\\saved_models", data_hora))
        os.makedirs(subcarpeta, exist_ok=True)

        # Guardar el model en un fitxer .h5
        model_path = os.path.join(subcarpeta, "model.h5")
        save_model(model_save, model_path)

        # Extreure els hiperparàmetres del model
        hiperparametres = model_save.get_config()

        # Guardar els hiperparàmetres en un fitxer .pkl
        hiperparametres_path = os.path.join(subcarpeta, "hiperparametres.pkl")

        # Guardar el fitxer CSV amb els resultats de l'estudi
        study_path = os.path.join(subcarpeta, "study.csv")
        study.trials_dataframe().to_csv(study_path)
        print(study.best_value)
        print(study.best_params)
        print(study.best_trial)

        with open(hiperparametres_path, 'wb') as f:
            pickle.dump(hiperparametres, f)

        # Copiar el fitxer CSV a la subcarpeta
        shutil.copy2(csv_path, subcarpeta)


    def search_and_train_with_optuna(self, n_searches, n_trials_per_search, model_save_path):
        """
        Realiza la búsqueda de los mejores modelos utilizando Optuna y entrena el mejor modelo.
        Guarda el modelo y repite el proceso n_searches veces.

        :param n_searches: Número de veces para repetir el proceso de búsqueda y entrenamiento.
        :param n_trials_per_search: Número de trials por búsqueda en Optuna.
        :param model_save_path: Ruta donde se guardarán los modelos entrenados.
        """
        study = None  # Inicializar el objeto de estudio como None
        best_params = None  # Inicializar los mejores parámetros como None
        for i in range(n_searches):
            print(f"\nBúsqueda y entrenamiento {i + 1} de {n_searches}")

            # Optimizar hiperparámetros con Optuna
            print("Optimizando hiperparámetros...")
            best_params, study = self.optimize(
                n_trials=n_trials_per_search, study=study, initial_params=best_params
            )
            print(f"Mejores hiperparámetros encontrados: {best_params}")

            # Guardar el mejor modelo
            save_path = os.path.join(model_save_path, f"best_model_{i + 1}.h5")
            self.guardar_model(self.best_model, study, self.input_file)
            print(f"Modelo guardado en {save_path}")


if __name__ == '__main__':

    prediccio_carpeta = False
    optuna_for = True
    usar_optuna = False
    prediccio = False
    guardar_model = False
    model_entrenat = None
    entrenar = False

    # Prediccions amb un model ja entrenat
    if prediccio_carpeta:
        directorio = r"C:\Users\u1050\PycharmProjects\RNN_LLE\for_model\Cat1_17042023\model\18042023__10_18"
        # directorio = r"C:\Users\u1050\PycharmProjects\RNN_LLE\for_model\Cat2_17042023\model\18042023__08_34"
        # directorio = r"C:\Users\u1050\PycharmProjects\RNN_LLE\for_model\Cat3_17042023\model\18042023__18_56"
        # directorio = r"C:\Users\u1050\PycharmProjects\RNN_LLE\for_model\Cat4_17042023\model\18042023__09_19"
        # directorio = r"C:\Users\u1050\PycharmProjects\RNN_LLE\for_model\Cat5_17042023\model\17042023__19_59"
        datos = None
        modelo = None
        hiperparametros = None

        for archivo in os.listdir(directorio):
            if archivo.endswith('.csv') and datos is None:
                fitxer = (os.path.join(directorio, archivo))
            elif archivo.endswith('.pkl') and hiperparametros is None:
                with open(os.path.join(directorio, archivo), 'rb') as f:
                    hiperparametros = pickle.load(f)
            elif archivo.endswith('.h5') and modelo is None:
                modelo = h5py.File(os.path.join(directorio, archivo), 'r')

        rnnFolder = RNNModel(fitxer)
        rnnFolder.best_model = load_model(modelo)
        rnnFolder.best_model.set_weights(hiperparametros)
        # Leer los datos
        data = rnnFolder.read_data()
        data_procesada = rnnFolder.preprocess_data()
        rnnFolder.split_data(data_procesada)
        # Ejecutar la predicción
        rnnFolder.predict_last_rows()

    else:
        input_file = r"./RNN_LLE/dades/Dades_Per_entrenar.csv"  # Reemplazar por la ruta del archivo CSV
        rnn = RNNModel(input_file)
        data = rnn.read_data()

        data_procesada = rnn.preprocess_data()
        rnn.split_data(data_procesada)
        if usar_optuna:
            print("Optimizando hiperparámetros...")
            best_params = rnn.optimize(n_trials=1)
            print(f"Mejores hiperparámetros encontrados: {best_params}")
            print(best_params)

            final_model = rnn.best_model
            print("Guardando el modelo...")
            final_model.save(rnn.fitxerModel)
            print(f"Modelo guardado en {rnn.fitxerModel}")

            if guardar_model:
                mySql = MysqlModel()
                mySql.guardar(final_model)
                # Leer los datos
                data = rnn.read_data(nomFitxer=".\\dades\\Dades_Per_entrenarN_P.csv")
                # Crear una instancia de la clase MysqlModel
                mysql_model = MysqlModel()
                # Obtener los datos del fichero de predicción
                fitxerModel = mysql_model.recuperar()
                print("Modelo guardado y recuperado de mysql")
            else:
                print("No se ha guardado el modelo en mysql")

        if optuna_for:
            n_searches = 1000
            n_trials_per_search = 1000
            model_save_path = ".\\saved_models"

            rnn.search_and_train_with_optuna(n_searches, n_trials_per_search, model_save_path)

        if prediccio:
            rnn.fitxerModel = ".\\saved_models\\best_model_1.h5"
            if model_entrenat:
                rnn.best_model = model_entrenat
            else:
                # Cargar el modelo
                rnn.best_model = load_model(rnn.fitxerModel)
            # Ejecutar la predicción
            rnn.predict_last_rows()
