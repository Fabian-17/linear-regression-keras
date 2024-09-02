import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# Obtiene los datos del archivo CSV
datos = pd.read_csv('altura_peso.csv', sep=",")
print(datos)

# Separa las variables de entrada y salida
x = datos['Altura'].values
y = datos['Peso'].values

# Normaliza los datos porque la red neuronal es sensible a la escala
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

# Inicialización del modelo
np.random.seed(2)
modelo = Sequential()

# Añade una capa densa
input_dim = 1
output_dim = 1
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# Define el optimizador y compilar el modelo
sgd = SGD(learning_rate=0.003)
modelo.compile(loss='mse', optimizer=sgd)

# Muestra el resumen del modelo
modelo.summary()

# Entrenamiento del modelo
num_epochs = 10000
batch_size = x.shape[0]
history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Obtener los parámetros w y b después del entrenamiento
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.4f}, b = {:.4f}'.format(w[0][0], b[0]))

# Grafica el error cuadrático medio (ECM) vs. épocas
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('ECM')
plt.title('ECM vs. Epochs')
plt.savefig('ecm_vs_epochs.png')  # Guardar la imagen
plt.close()

# Grafica los datos originales y la recta de regresión
y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Datos Originales')
plt.plot(x, y_regr, 'r', label='Recta de Regresión')
plt.title('Datos y Regresión Lineal')
plt.legend()
plt.savefig('datos_regresion.png')  # Guardar la imagen
plt.close()

# Predicción para una altura específica
altura_pred = np.array([170])
altura_pred = (altura_pred - x_mean) / x_std  # Normaliza
peso_pred = modelo.predict(altura_pred)
peso_pred = peso_pred * y_std + y_mean  # Desnormaliza

print("El peso predicho para una persona con una altura de {} cm es {:.1f} kg".format(170, peso_pred[0][0]))