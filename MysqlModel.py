import pymysql
import os

import datetime

import optuna
import matplotlib.pyplot as plt


class MysqlModel:

    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'root'
        self.database = 'bestmodels'
        self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password,
                                          database=self.database, connect_timeout=120)
        self.nomFitxer = '.\\model\\model.h5'

    def guardar(self, model):
        model.save(self.nomFitxer)
        # Comprobar si el archivo h5 existe
        if not os.path.isfile(self.nomFitxer):
            print("El archivo h5 proporcionado no es válido")
            return

        # Obtener la fecha y hora actuales
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        # Leer el archivo h5 como datos binarios
        with open(self.nomFitxer, 'rb') as file:
            h5_data = file.read()

        # Conectarse a la base de datos MySQL
        try:
            # Establecer la conexión con la base de datos
             with self.connection.cursor() as cursor:
                # Insertar el archivo h5 y la fecha y hora actuales en la tabla
                sql = "INSERT INTO bestmodels.models (data_model, fitxer_model) VALUES (%s, %s)"
                cursor.execute(sql, (current_datetime, h5_data))

                # Confirmar la transacción
                self.connection.commit()
        finally:
            # Cerrar la conexión a la base de datos
            self.connection.close()

    def recuperar(self):
        # Conectarse a la base de datos MySQL

        try:
            with self.connection.cursor() as cursor:
                # Obtener el registro más reciente de la tabla rnn_models
                sql = "SELECT id, data_model, fitxer_model FROM bestmodels.models ORDER BY id DESC LIMIT 1"
                cursor.execute(sql)
                result = cursor.fetchone()

                if result is None:
                    print("No se encontraron registros en la base de datos.")
                    return

                id_model, model_name, model_file = result

                with open(self.nomFitxer, 'wb') as file:
                    file.write(model_file)

                print(f"El archivo h5 más reciente se ha guardado en: {self.nomFitxer}")
                return self.nomFitxer
        finally:
            # Cerrar la conexión a la base de datos
            self.connection.close()
