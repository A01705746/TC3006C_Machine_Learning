'''
  Algoritmo de regresion lineal sin manejo de frameworks 
  Enrique Santos Fraire - A01705746
  26/08/2022
'''

from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carga y lectura del data set
df = pd.read_excel('Raisin_Dataset.xlsx')

# Escalamiento de nuestras muestras para no desbrodar el modelo
df['MaAL'] = df['MajorAxisLength']/1000
df['MiAL'] = df['MinorAxisLength']/1000

# Asignacion de muestras x, y, parámetros del tamaño de la muestra inicializados en 0
samples = df[["MaAL", "MiAL"]].to_numpy().tolist()
y = df['Eccentricity'].to_numpy().tolist()
params = np.zeros(len(samples[0])+1).tolist()

# Separación de los datos muestra y y en train y test
samples_train, samples_test, y_train, y_test = train_test_split(samples, y, random_state=1)

# Se inicializa el arreglo de errores, el learning rate y las épocas
__errors__ = []
alfa = 0.1
epochs = 0

'''
    Hipótesis de función linear h(x)

    Args:
      params - lista de parámetros o thetas correspondiente a cada muestra o x
      sample - lista de muestras o x

    Returns:
      Valor de la hipótesis
'''
def hyp (params, sample):
  acum = 0
  for i in range(len(params)):
    acum = acum + params[i]*sample[i]
  return acum

'''
    Gradiente Descendiente, actualiza los parámetros

    Args:
      params - lista de parámetros o thetas correspondiente a cada muestra o x
      samples - lista multidimensional que contiene todas las muestras o x
      y - lista de valores reales
      alfa - learning rate

    Returns:
      Lista de nuevos parámetros
'''
def gd (params, samples, y, alfa):
  temp = []
  for i in range (len(params)):
    acum = 0
    for j in range (len(samples)):
      error = (hyp(params, samples[j]) - y[j])
      acum = acum + error * samples[j][i]
    temp.append( params[i] - alfa/(len(samples)) * acum )
  return temp

'''
    Guarda el error de la hipótesis estimada con y

    Args:
      params - lista de parámetros o thetas correspondiente a cada muestra o x
      samples - lista multidimensional que contiene todas las muestras o x
      y - lista de valores reales
'''
def errors (params, samples, y):
  global __errors__
  acum = 0

  for i in range (len(samples)):
    h = hyp(params, samples[i])
    print( 'hyp: %f  y: %f'  % (h, y[i]))
    error = (h - y[i]) ** 2
    acum = acum + error
  mean_error = acum / len(samples)
  __errors__.append(mean_error)

'''
    Guarda el error de diferencia entre la "y" real y la "y" del modelo

    Args:
      y_pred - lista de valores del modelo
      y_test - lista de valores reales
'''
def errors_test (y_test, y_pred):
  acum = 0
  for i in range(len(y_test)):
    acum = acum + (y_test[i] - y_pred[i]) ** 2
  return acum

'''
    Se le añade un 1 a las muestras train de x para facilitar
    las operaciones con los parámetros
'''
for i in range(len(samples_train)):
	if isinstance(samples_train[i], list):
		samples_train[i]=  [1] + samples_train[i]
	else:
		samples_train[i]=  [1, samples_train[i]]

'''
    Corrida del modelo de regresión
'''
while True:
  oldparams = params
  params = gd (params, samples_train, y_train, alfa)
  errors (params, samples_train, y_train)
  print (params)
  epochs += 1
  if (oldparams == params or epochs == 1000):
    print ("samples_train")
    print (samples_train)
    
    print ("params")
    print (params)

    print ("error")
    print (__errors__[epochs-1])

    break

'''
    Se le añade un 1 a las muestras test de x para facilitar
    las operaciones con los parámetros
'''
for i in range(len(samples_test)):
	if isinstance(samples_test[i], list):
		samples_test[i]=  [1] + samples_test[i]
	else:
		samples_test[i]=  [1, samples_test[i]]

'''
    Se obtiene las y de predicción del modelo
'''
y_pred = [np.dot(x, params)for x in samples_test]

# Impresión del error
print ("error_test")
print(errors_test(y_test, y_pred))

plt.figure(figsize = (8, 6))

# Figura de comportamiento del error por época
plt.subplot(2, 1, 1)
plt.plot(__errors__)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error adjustment")

# Comportamiento del error de y_test con y_train
plt.subplot(2, 1, 2)
plt.scatter(range(len(y_test)), y_test, label = 'y_test')
plt.scatter(range(len(y_test)), y_pred, label = 'y_pred')
# plt.plot([x-y for x, y in zip(y_test, y_pred)])
plt.ylabel("y value")
plt.xlabel("Position")
plt.legend()
plt.title("Test data vs predicted data")

plt.show()