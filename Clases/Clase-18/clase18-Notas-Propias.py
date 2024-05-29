# Varianza: Cuanto se desvian los datos de la media o del promedio
# Por ejemplo, si tenemos los datos [1, 2, 3, 4, 5] la media es 3 y la varianza es 2 ya que, calculando,
# (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 2

def calcular_varianza(datos):
    media = sum(datos) / len(datos)
    varianza = sum([(dato - media)**2 for dato in datos]) / len(datos)
    return varianza

def calcular_media (datos):
    return sum(datos) / len(datos)

# El producto interno de dos vectores es igual a la suma de los productos de sus componentes
# Por ejemplo, el producto interno de los vectores [1, 2, 3] y [4, 5, 6] es 1*4 + 2*5 + 3*6 = 32

def producto_interno(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])

# Covarianza: Es una medida de la relación entre dos variables. Por ejemplo
# si tenemos dos variables x e y, la covarianza es la media de los productos de las diferencias de cada variable con su media
# Ej, si tenemos los datos [1, 2, 3, 4, 5] y [2, 3, 4, 5, 6] la covarianza es 1 ya que, calculando
# (1-3)*(2-4) + (2-3)*(3-4) + (3-3)*(4-4) + (4-3)*(5-4) + (5-3)*(6-4) = 1

# Si la covarianza es positiva, significa que las variables tienden a aumentar juntas
# Si la covarianza es negativa, significa que una variable tiende a aumentar mientras la otra disminuye
# Si la covarianza es 0, significa que las variables son independientes

def calcular_covarianza(datos1, datos2):
    media1 = sum(datos1) / len(datos1)
    media2 = sum(datos2) / len(datos2)
    covarianza = sum([(datos1[i] - media1) * (datos2[i] - media2) for i in range(len(datos1))]) / len(datos1)
    return covarianza

# Correlación: Es una medida de la relación entre dos variables, pero a diferencia de la covarianza, la correlación
# es una medida normalizada que siempre está entre -1 y 1. La correlación es igual a la covarianza dividida por el producto
# de las desviaciones estándar de las dos variables

def calcular_correlacion(datos1, datos2):
    covarianza = calcular_covarianza(datos1, datos2)
    desviacion1 = calcular_varianza(datos1)**0.5
    desviacion2 = calcular_varianza(datos2)**0.5
    correlacion = covarianza / (desviacion1 * desviacion2)
    return correlacion

# La matriz de covarianza de un conjunto de datos es una matriz cuadrada que contiene las covarianzas entre todas las
# combinaciones de variables. Por ejemplo, si tenemos los datos [[1, 2, 3], [4, 5, 6], [7, 8, 9]] la matriz de covarianza
# es [[varianza([1, 4, 7]), covarianza([1, 4, 7], [2, 5, 8]), covarianza([1, 4, 7], [3, 6, 9])],
# [covarianza([2, 5, 8], [1, 4, 7]), varianza([2, 5, 8]), covarianza([2, 5, 8], [3, 6, 9])],
# [covarianza([3, 6, 9], [1, 4, 7]), covarianza([3, 6, 9], [2, 5, 8]), varianza([3, 6, 9])]]
# Donde varianza y covarianza son las funciones definidas anteriormente