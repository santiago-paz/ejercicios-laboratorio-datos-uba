"""
Desarrollado para la materia Laboratorio de Datos dictada por el Instituto de Calculo de la Facultad de Ciencias
Exactas y Naturales de la Universida de Buenos Aires durante el Primer Cuatrimestre de 2024.
"""

import logging
import os
from itertools import chain

import matplotlib.animation
import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import keras


class PathRecorder(keras.callbacks.Callback):
    """
    Un callback para registrar los pesos y el intercept al final de cada epoca, para luego poder graficar
    """
    def on_epoch_end(self, epoch, logs=None):
        self.model.path.append(np.array(list(chain([self.model.b.numpy()], self.model.w.numpy()))))


class Regressor(tf.keras.Model):
    """
    Clase para el regresor basado en TensorFlow y Keras. Hereda de la clase Model de Keras (
    documentacion : https://keras.io/api/models/model_training_apis/ )
    Tiene los siguientes atributos:
    w : pesos (tensor de TensorFlow)
    b : intercept (tensor de TensorFlow)
    with_intercept : indica si f tiene intercept ( bool )
    f : la funcion con la cual se quiere hacer regresion ( function )
    opt : tipo de optimizador ('gd' : descenso por gradiente; 'sgd' : descenso por gradiente estocastico)
    path : lista con los pesos al final de cada epoca ( list[tensor] )
    hist : guarda el objeto History que devuelve el metodo fit de keras.Model
    loss_w : funcion de perdida como funcion de los pesos y del bias ( L(b,w) )
    classifying : indica si se trata de un problema de clasificacion ( bool )
    """

    def __init__(self, **kwargs):
        """ Inicializa la instancia de la clase. Recibe los mismos argumentos que keras.Model """
        super().__init__(**kwargs)
        self.w = None
        self.b = None
        self.with_intercept = False
        self.f = None
        self.opt = None
        self.path = []
        self.hist = None
        self.loss_w = None
        self.classifying = False

    @property
    def weights_(self):
        """ Devuelve los pesos como un vector de numpy """
        return self.w.numpy()

    @property
    def bias_(self):
        """ Devuelve el bias como un float """
        return self.b.numpy()

    def add_f(self, f, dim_w=None, w0=None, b0=None, random_state=None, opt='sgd'):
        """
        Configura la funcion con la que se realizara la regresion.
        :param f: funcion con la que se realizara la regresion. Debe tomar tres argumentos en este orden: x, w,
        b o dos argumentos en este orden: x, w
        :type f: function
        :param dim_w: dimension de los pesos, en caso de querer generarlos aleatoriamente. dim_w o w0 debe ser
        especificado.
        :type dim_w: int
        :param w0: peso inicial. dim_w o w0 debe ser especificado.
        :type w0: numpy.ndarray
        :param b0: bias inicial, en caso de querer especificarlo
        :type b0: int | float
        :param random_state: semilla para la generacion aleatoria de pesos y bias y seleccion de conjunto de validacion
        :type random_state: int
        :param opt: tipo de optimizador ('gd' o 'sgd', 'sgd' por defecto)
        :type opt: str
        """

        if random_state is not None:
            keras.utils.set_random_seed(random_state)
            # np.random.seed(random_state)

        # Configura los pesos dados por w0 o genera pesos aleatoriamente segun dim_w
        try:
            self.w = keras.Variable(w0.astype('float')) if w0 is not None else keras.Variable(np.random.randn(dim_w))
        except TypeError:   # Ambos son None
            raise TypeError("Debe especificar la dimension de w (dim_w) o proveer un array de pesos iniciales (w0)")
        except AttributeError:  # w0 es un escalar
            self.w = keras.Variable([float(w0)])

        # Si f tiene 3 argumentos, hay intercept
        if f.__code__.co_argcount == 3:
            self.with_intercept = True
            self.f = f
        else:
            self.f = lambda x, w, b: f(x, w) + b

        # Configura el bias
        self.b = keras.Variable(float(b0)) if b0 is not None else keras.Variable(np.random.rand())

        self.opt = opt.lower()
        if opt not in ['sgd', 'gd']:
            raise ValueError("opt debe ser 'gd' para descenso por gradiente y 'sgd' para descenso estocastico")

    def call(self, x):
        """ Define que devuelve el modelo al ser evaluado en x """
        return self.f(x, self.w, self.b)

    def fit_(self, X, y, epochs=1000, learning_rate=0.05, validation_split=0.0, validation_data=None, batch_size=1,
             verbose=1, early_stopping=False, patience=100, start_from_epoch=0, lr_scheduler=None, classify=False,
             clipnorm=None):
        """
        Realiza el ajuste
        :param X: datos de las variables predictoras
        :type X: numpy.ndarray | pandas.DataFrame | pandas.Series
        :param y: datos de la variable dependiente
        :type y: numpy.ndarray | pandas.Series
        :param epochs: cantidad de epocas
        :type epochs: int
        :param learning_rate: learning_rate
        :type learning_rate: int | float
        :param validation_split: porcentaje de los datos de X que seran usados como validacion (utilizar 0.0 para no
        separar conjunto de validacion)
        :type validation_split: float
        :param validation_data: datos que son usados para validacion. Esto se utiliza si ya determinamos
        anteriormente un conjunto de validacion.
        :type validation_data: numpy.ndarray | pandas.DataFrame | pandas.Series
        un valor distinto de 0.0 en validation_split
        :param batch_size: tamaño del batch. En especial, se puede usar len(y) para hacer descenso por gradiente y 1
        para desnso por gradiente estocastico tradicional. El valor debe estar en el intervalo [1, len(y)]
        :type batch_size: int
        :param verbose: si queremos que se imprima en pantalla el progreso de entrenamiento (1: si, 0: no)
        :type verbose: int
        :param early_stopping: si queremos activar Early Stopping
        :type early_stopping: bool
        :param patience: pasada enta cantidad de epocas sin que mejore el error, Early Stopping detiene el entrenamiento
        :type patience: int
        :param start_from_epoch: epoca a partir de la cual se activa Early Stopping
        :type start_from_epoch: int
        :param lr_scheduler: funcion que modifica el learning_rate a traves de las epocas. Debe ser una funcion que
        reciba los argumentos (epoch, learning_rate)
        :type lr_scheduler: function
        :param classify: indica si se trata de un problema de clasificacion
        :type classify: bool
        :param clipnorm: valor al cual limitar la norma del gradiente. Como el gradiente es una direccion,
        podemos tomar cualquier vector en esa direccion, de menor norma para evitar que el algoritmo diverja. Por
        ejemplo, si clipnorm = 1, cada vez que el gradiente tenga norma mayor que 1, se lo normaliza.
        :type clipnorm: int | float
        """

        # Si X, y no son arrays de numpy, los convertimos
        if not isinstance(X, np.ndarray):
            x = X.to_numpy()
        else:
            x = X
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        # Definimos la funcion de perdida (MSE para regresion, CEE para clasificacion)
        loss = (tf.keras.losses.MeanSquaredError if not classify
                else tf.keras.losses.BinaryCrossentropy(from_logits=False))
        # Definimos las matricas para el caso de clasificacion
        metrics = None if not classify else [keras.metrics.BinaryAccuracy(), keras.metrics.BinaryCrossentropy()]
        self.classifying = classify

        # Compilamos el modelo
        self.compile(
            run_eagerly=False,
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                              clipnorm=clipnorm),   # Configuramos SGD
            loss=loss,
            metrics=metrics
        )

        batch_size = batch_size if self.opt == 'sgd' else len(y)

        # Definimos el conjunto de validacion. Si bien TensorFlow realiza una eleccion automatica de conjunto de
        # validacion, elige siempre los ultimos % datos del dataset. Si el dataset esta ordenado, esto podria
        # resultar problematico. Por eso en estas lineas realizamos una eleccion aleatoria de los datos de validacion
        # y se los pasamos al metodo fit.
        val_index = np.array([])
        if validation_split > 0 and validation_data is None:
            # Separa aleatoriamente un conjunto de validacion
            val_index = np.random.choice(len(y), int(len(y) * validation_split), replace=False)
            validation_data = (x[val_index], y[val_index])
            self.loss_w = lambda w, b: np.mean((y - self.f(x, w, b))**2,
                                               where=np.array([i in val_index for i in range(len(y))]))
        else:
            self.loss_w = lambda w, b: np.mean((y - self.f(x, w, b))**2)

        # Definimos los callbacks
        callbacks = []
        if early_stopping:
            metric = 'val_loss' if validation_split > 0 else 'loss'
            callbacks.append(keras.callbacks.EarlyStopping(monitor=metric,
                                                           patience=patience,
                                                           restore_best_weights=True,
                                                           start_from_epoch=start_from_epoch,
                                                           mode='min')
                             )
        if lr_scheduler is not None:
            callbacks.append(keras.callbacks.LearningRateScheduler(lr_scheduler))

        callbacks.append(PathRecorder())

        # Entrenamos el modelo
        self.hist = self.fit(x, y,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=validation_data,
                             verbose=verbose,
                             callbacks=callbacks)

    def plot_path(self):
        """ Grafica el camino que realiza el algoritmo, marcando el valor de los pesos (o del peso y el bias) al
        final de cada epoca. Solo funciona si hay bias y la dimension de w es 1 o si no hay bias y la dimension de w es
        2. """

        if len(self.w.numpy()) >= 3 - self.with_intercept:
            raise ValueError("La dimension de w es demasiado grande para graficar en el plano.")

        # Con esto marcamos los limites del grafico para observar todo el recorrido del algoritmo
        i = 0 if self.with_intercept else 1
        last_w = self.path[-1][i:]

        max_dist = max(norm(s[i:] - last_w, np.inf) for s in self.path)
        left, right = (last_w[0] - max_dist*1.1), (last_w[0] + max_dist*1.1)
        bottom, top = (last_w[1] - max_dist*1.1), (last_w[1] + max_dist*1.1)

        x_axis = np.linspace(left, right)
        y_axis = np.linspace(bottom, top)
        X, Y = np.meshgrid(x_axis, y_axis)

        # Con esto generamos el mapa de calor
        if self.with_intercept:
            Z = np.fromiter(map(self.loss_w, Y.ravel(), X.ravel()), X.dtype).reshape(X.shape)
        else:
            Z = np.fromiter(map(self.loss_w, zip(X.ravel(), Y.ravel()), np.zeros(X.size)), X.dtype).reshape(X.shape)

        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                       cmap=cm.plasma, extent=(left, right, bottom, top))
        fig.colorbar(im)

        # Con esto generamos la linea del trayecto del algoritmo
        ax.add_line(lines.Line2D(
            [w[0+i] for w in self.path],
            [w[1+i] for w in self.path],
            color='r', lw=2, marker='o', label='Recorrido del algoritmo'
        ))

        # Agregamos etiquetas y titulos
        if self.with_intercept:
            ax.set_xlabel('$b$')
            ax.set_ylabel('$w_0$')
        else:
            ax.set_xlabel('$w_1$')
            ax.set_ylabel('$w_0$')

        ax.set_title('Recorrido del algoritmo')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def plot_loss(self):
        """ Grafica la evolucion del error a lo largo de las epocas. """

        loss_abbv = 'MSE' if not self.classifying else 'CEE'

        epochs = np.arange(1, len(self.hist.history['loss'])+1)
        loss = np.array(self.hist.history['loss'])
        plot = (so.Plot(x=epochs)
                .add(so.Line(color='blue'), y=loss, label=f'Training {loss_abbv}')
                )

        # Agregamos la evolucion del error de validacion si hubo conjunto de validacion
        try:
            val_loss = np.array(self.hist.history['val_loss'])
            plot = plot.add(so.Line(color='orange'), y=val_loss, label=f'Validation {loss_abbv}')
        except KeyError:    # No hay validacion
            pass

        title = 'Mean Squared Error' if not self.classifying else 'Cross-Entropy Error'
        plot = plot.label(title=title, x='Epoch', y=loss_abbv)
        plot.show()

    def animate_regression(self, X, y):
        """ Anima el grafico de regresion a traves de las epocas. Solo funciona si los datos de X tienen uno o dos
        features. """

        if X.ndim >= 2:
            raise ValueError("La dimension de X es demasiado grande para graficar en el plano.")

        # Definimos los limites de grafico
        fig, ax = plt.subplots()
        left, right = X.min()*0.95 - 0.05*(np.isclose(X.min(), 0)), X.max()*1.05
        bottom, top = y.min()*0.95 - 0.05*(np.isclose(y.min(), 0)), y.max()*1.05
        plt.xlim(left, right)
        plt.ylim(bottom, top)

        if isinstance(X, pd.Series):
            ax.set_xlabel(X.name)
        if isinstance(y, pd.Series):
            ax.set_ylabel(y.name)

        # Graficamos los datos y la regresion correspondiente a los pesos y bias iniciales.
        x_arr = np.linspace(left, right)
        sns.scatterplot(x=X, y=y)
        line = ax.plot(x_arr,
                       self.f(x_arr, self.path[0][1:], self.path[0][0]),
                       color='r',
                       lw=2
                       )[0]

        # Agregamos texto para indicar el valor de los pesos y del bias
        text_str = f'$w= $ {np.round(self.path[0][1:], 2)} \n $b =$ {self.path[0][0]:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        text_artist = ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                              verticalalignment='top', bbox=props)

        # Definimos una funcion que determina cada fotograma de la animacion
        def update(i):
            # Actualizamos el grafico de la funcion de regresion
            line.set_ydata(self.f(x_arr, self.path[i][1:], self.path[i][0]))
            # Actualizamos el titulo del grafico
            ax.set_title(f'Epoch {i}')
            # Actualizamos el texto
            text_artist.set_text(f'$w= $ {np.round(self.path[i][1:], 2)} \n $b =$ {self.path[i][0]:.2f}')
            return line

        # Generamos la animacion
        # Mas informacion en : https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
        animation = matplotlib.animation.FuncAnimation(fig, update,
                                                       frames=len(self.path),
                                                       repeat=False)
        plt.show()

        return animation


def scale_center_train_test(full, train, test, center=True):
    """
    Dados datos los datos completos (X o y), el conjunto de entrenamiento y el conjunto de testeo, los normaliza y
    los centra.
    :param full: conjuntos de datos (todos)
    :type full: pd.Series | pd.DataFrame
    :param train: conjunto de entrenamiento
    :type train: pd.Series | pd.DataFrame
    :param test: conjunto de testeo
    :type test: pd.Series | pd.DataFrame
    :param center: indica si se deben centrar o no los datos alrededor de la media
    :type center: bool
    :return: datos de entrenamiento y testeo escalados y centrados
    :rtype: pd.Series | pd.DataFrame
    """

    # Si los datos vienen como un pandas.Series los convertimos a DataFrame para poder usar MinMaxScaler
    if isinstance(full, pd.Series):
        train = pd.DataFrame(train, columns=[full.name])
        test = pd.DataFrame(test, columns=[full.name])

    # Escalamos segun el conjunto de entrenamiento
    scaler = MinMaxScaler().set_output(transform='pandas')
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Centramos
    if center:
        train_mean = train_scaled.mean()
        train_scaled -= train_mean
        test_scaled -= train_mean

    # Si antes convertimos Series en DataFrame, volvemos a Series
    if isinstance(full, pd.Series):
        train_scaled = train_scaled[full.name]
        test_scaled = test_scaled[full.name]

    return train_scaled, test_scaled


def train_test_split_scale_center(X, y, transform_y=False, center=True, **kwargs):
    """
    Escala y centra los datos X e y.
    :param X: datos de la(s) variable(s) predictora(s)
    :type X: pd.Series | pd.DataFrame
    :param y: datos de la variable dependiente
    :type y: pd.Series
    :param transform_y: indica si se le aplica la transformacion a los datos de y
    :type transform_y: bool
    :param center: indica si se centran los datos
    :type center: bool
    :param kwargs: argumentos para la funcion train_test_split de scikit-learn
    :return: conjuntos de entrenamiento y testeo escalados y )opcionalmente) centrados
    :rtype: tuple[pd.Series | pd.DataFrame]
    """
    # Si y no es un pandas.Series, la transforma a pandas.Series
    if isinstance(y, pd.DataFrame):
        y = y[y.columns[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    X_train_scaled, X_test_scaled = scale_center_train_test(X, X_train, X_test, center)


    if transform_y:
        y_train_scaled, y_test_scaled = scale_center_train_test(y, y_train, y_test, False)
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    return X_train_scaled, X_test_scaled, y_train, y_test
