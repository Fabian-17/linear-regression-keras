# Proyecto de Predicción de Peso Basado en la Altura usando una Red Neuronal Simple

Este proyecto implementa un modelo de regresión lineal utilizando redes neuronales para predecir el peso de una persona a partir de su altura. Se emplea una red neuronal simple con una capa densa entrenada usando el algoritmo de optimización `Stochastic Gradient Descent` (SGD). Los datos de entrenamiento provienen de un archivo CSV que contiene pares de altura y peso.


## Requerimientos del Proyecto

### Dependencias

Asegúrate de instalar las siguientes dependencias antes de ejecutar el código:

- `Python 3.x`: Si no lo tienes instalado, puedes descargarlo desde [python.org](https://www.python.org/downloads/).
- `pandas`
- `numpy`
- `matplotlib`
- `keras`
- `tensorflow`

### Configuración del Entorno Virtual

Se recomienda usar entornos virtuales para evitar conflictos entre las dependencias de diferentes proyectos. Sigue estos pasos para crear y activar un entorno virtual:

```bash
# Instalar la herramienta virtualenv si no está instalada
pip install virtualenv

# Crear un nuevo entorno virtual
virtualenv venv

# Activar el entorno virtual (Windows)
venv\Scripts\activate
# o (Linux/macOS)
source venv/bin/activate
```

Puedes instalar estas dependencias ejecutando el siguiente comando:

```bash
pip install pandas numpy matplotlib keras tensorflow
```

## Explicación del Código

### Carga y Normalización de los Datos

Los datos se cargan desde el archivo `altura_peso.csv`, y las variables de entrada (`Altura`) y salida (`Peso`) se normalizan para mejorar el rendimiento de la red neuronal. La normalización es importante porque las redes neuronales son sensibles a la escala de los datos. Aquí, tanto la altura como el peso se escalan restando la media y dividiendo por la desviación estándar.

```bash
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)
x = (x - x_mean) / x_std
y = (y - y_mean) / y_std
```

### Creación del Modelo

Se utiliza una red neuronal muy simple con una sola capa densa que tiene una función de activación lineal para ajustar los datos de altura y peso.

```bash
modelo = Sequential()
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))
```

### Entrenamiento del Modelo

El modelo se entrena utilizando el algoritmo de optimización `SGD` con una tasa de aprendizaje baja para ajustar la relación entre altura y peso.

```bash
sgd = SGD(learning_rate=0.003)
modelo.compile(loss='mse', optimizer=sgd)
history = modelo.fit(x, y, epochs=10000, batch_size=batch_size, verbose=1)
```

### Generación de Gráficos

Debido a que el entorno en el que se ejecuta el código puede no permitir la visualización interactiva de gráficos, los gráficos generados (el gráfico de pérdida y la recta de regresión) se guardan como imágenes (`.png`) utilizando `plt.savefig()`. Esto asegura que los gráficos puedan ser revisados posteriormente, incluso si el entorno no los muestra en tiempo real.

```bash
plt.savefig('ecm_vs_epochs.png')
plt.savefig('datos_regresion.png')
```

### Uso de Jupyter Notebook

Se incluye un archivo Jupyter Notebook (`prueba.ipynb`) para ejecutar el código en un entorno interactivo, como JupyterLab. Este entorno facilita la visualización de gráficos sin necesidad de guardarlos como imágenes, ya que los gráficos se muestran automáticamente en las celdas.

## Ejecución del Proyecto

Para ejecutar el código y entrenar el modelo, simplemente ejecuta el archivo `main.py`:

```bash
python main.py
```

o para sistemas operativos de Linux/MacOS

```bash
python3 main.py
```

## Predicciones

El modelo puede predecir el peso de una persona para una altura dada. Por ejemplo, la predicción para una persona con una altura de 170 cm se realiza después de desnormalizar el resultado obtenido por la red:

```bash
peso_pred = modelo.predict(altura_pred)
peso_pred = peso_pred * y_std + y_mean  # Desnormaliza el peso
```

El peso predicho se imprimirá en la consola.