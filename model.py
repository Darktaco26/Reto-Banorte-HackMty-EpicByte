import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def plot(time, series, format="-", start= 0, end = None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])


dt = pd.read_csv('INP_INP.csv')
inpc_data = []
sub_data = []
not_data = []
merc_data = []
serv_data = []
agro_data = []
energ_data = []

indices_data = []
inpc = dt["INPC"]
subyacente = dt["Subyacente"]
mercancias = dt["Mercancìas"]
servicios = dt["Servicios"]
noSubyacente = dt["No subyacente"]
agropecuarios = dt["Agropecuarios"]
energeticos = dt["Energéticos"]

for n in range(480):
    inpc_data.append(inpc[n:n+24])
    sub_data.append(subyacente[n:n+24])
    not_data.append(noSubyacente[n:n+24])
    merc_data.append(mercancias[n:n + 24])
    serv_data.append(servicios[n:n + 24])
    agro_data.append(agropecuarios[n:n + 24])
    energ_data.append(energeticos[n:n + 24])
#    indices_data.append(indices[n+11:n+24])

inpc_data = np.array(inpc_data)
sub_data = np.array(sub_data)
not_data = np.array(not_data)
merc_data = np.array(merc_data)
serv_data = np.array(serv_data)
agro_data = np.array(agro_data)
energ_data = np.array(energ_data)
# indices_data = np.array(not_data)
inpc = np.array(inpc[24:]).reshape((480,1))
subyacente = np.array(subyacente[24:]).reshape((480,1))
mercancias = np.array(mercancias[24:]).reshape((480,1))
servicios = np.array(servicios[24:]).reshape((480,1))
noSubyacente = np.array(noSubyacente[24:]).reshape((480,1))
agropecuarios = np.array(agropecuarios[24:]).reshape((480,1))
energeticos = np.array(energeticos[24:]).reshape((480,1))

print(inpc.shape)
print(inpc_data.shape)

temp_steps = []
cont = 0
for row in inpc:
    temp_steps.append(cont)
    cont += 1
'''
input_inpc = Input(shape=(inpc_data[0].shape));
dense_1 = Dense(10, activation='relu')(input_inpc);
dense_1 = Dense(1)(dense_1)
input_sub_data = Input(shape=(sub_data[0].shape));
dense_2 = Dense(10, activation='relu')(input_sub_data);
dense_3 = Dense(1)(dense_2)
input_not_data = Input(shape=(not_data[0].shape));
dense_not = Dense(10, activation='relu')(input_not_data);
dense_5 = Dense(1)(dense_not)
model = Model(inputs=[input_inpc, input_sub_data, input_not_data], outputs=[dense_1, dense_3, dense_5]);
'''
input_merca = Input(shape=(merc_data[0].shape)); #input merca
dense_merca = Dense(20, activation='relu')(input_merca);
dropout_merca = tf.keras.layers.Dropout(0.4)(dense_merca)
dense_merca2 = Dense(10, activation='relu')(dropout_merca);
dense_2 = Dense(1)(dense_merca2) #Output de la prediccion de merca

input_servicios = Input(shape=(serv_data[0].shape)); #input servicios
dense_servicios = Dense(20, activation='relu')(input_servicios);
dropout_servicios = tf.keras.layers.Dropout(0.4)(dense_servicios)
dense_servicios2 = Dense(10, activation='relu')(dropout_servicios);
dense_5 = Dense(1)(dense_servicios2) #Output de la prediccion de servicios

input_agrupe = Input(shape=(agro_data[0].shape)); #input agrupe
dense_agrupe = Dense(20, activation='relu')(input_agrupe);
dropout_agrupe = tf.keras.layers.Dropout(0.4)(dense_agrupe)
dense_agrupe2 = Dense(10, activation='relu')(dropout_agrupe);
dense_8 = Dense(1)(dense_agrupe2) #Output de la prediccion de agrupe

input_energ = Input(shape=(energ_data[0].shape)); #input energ
dense_energ = Dense(20, activation='relu')(input_energ);
dropout_energ = tf.keras.layers.Dropout(0.4)(dense_energ)
dense_energ2 = Dense(10, activation='relu')(dropout_energ);
dense_11 = Dense(1)(dense_energ2) #Output de la prediccion de energeticos

input_sub_data = Input(shape=(sub_data[0].shape)); #input subyacente
dense_subya = Dense(10, activation='relu')(input_sub_data);
concatenate_subyacente = tf.keras.layers.Concatenate(axis=-1)([dense_merca2, dense_servicios2, dense_subya])
dense_subya2 = Dense(10, activation='relu')(concatenate_subyacente)
dense_14 = Dense(1)(dense_subya2) #Output de la prediccion de subyacente

input_not_data = Input(shape=(not_data[0].shape)); #input no subyacente
dense_not = Dense(10, activation='relu')(input_not_data);
concatenate_notsubyacente = tf.keras.layers.Concatenate(axis=-1)([dense_agrupe2, dense_energ2, dense_not])
dense_not2 = Dense(10, activation='relu')(concatenate_notsubyacente)
dense_17 = Dense(1)(dense_not2) #Output de la prediccion de no subyacente


input_inpc = Input(shape=(inpc_data[0].shape)); #input inpc
dense_1 = Dense(12, activation='relu')(input_inpc);
concatenate_inpc = tf.keras.layers.Concatenate(axis=-1)([dense_subya2, dense_not2, dense_1])
dense_inpc = Dense(10, activation='relu')(concatenate_inpc)
dense_20 = Dense(1)(dense_inpc) #Output de la prediccion de INPC

model = Model(inputs=[input_merca, input_servicios, input_agrupe, input_energ, input_sub_data, input_not_data, input_inpc],
              outputs=[dense_2, dense_5, dense_8, dense_11, dense_14, dense_17, dense_20]);

model.compile(optimizer='Adam',
              loss = {'dense_2': 'mae',
                      'dense_5': 'mae',
                      'dense_8': 'mae',
                      'dense_11': 'mae',
                      'dense_14': 'mae',
                      'dense_17':'mae',
                      'dense_20': 'mae'},
              metrics= {'dense_2': tf.keras.metrics.RootMeanSquaredError(),
                      'dense_5': tf.keras.metrics.RootMeanSquaredError(),
                      'dense_8': tf.keras.metrics.RootMeanSquaredError(),
                      'dense_11': tf.keras.metrics.RootMeanSquaredError(),
                        'dense_14': tf.keras.metrics.RootMeanSquaredError(),
                        'dense_17': tf.keras.metrics.RootMeanSquaredError(),
                        'dense_20': tf.keras.metrics.RootMeanSquaredError(),
                        }
              )
model.fit([merc_data, serv_data, agro_data, energ_data, sub_data, not_data, inpc_data],
          [mercancias, servicios, agropecuarios, energeticos, subyacente, noSubyacente, inpc], epochs=600, batch_size=20)

p_mercancias, p_servicios, p_agropecuarios, p_energia, predict_sub, predict_not, predict_incp = model.predict([merc_data, serv_data, agro_data, energ_data, sub_data, not_data, inpc_data])



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(temp_steps, inpc, 'tab:red')
axs[0, 0].plot(temp_steps, predict_incp, 'tab:blue')
axs[0, 0].set_title('INPC (2001-2020)')
axs[0, 1].plot(temp_steps, subyacente, 'tab:red')
axs[0, 1].plot(temp_steps, predict_sub, 'tab:blue')
axs[0, 1].set_title('Subyacente (2001-2020)')
axs[1, 0].plot(temp_steps, noSubyacente, 'tab:red')
axs[1, 0].plot(temp_steps, predict_not, 'tab:blue')
axs[1, 0].set_title('No Subyacente (2001-2020)')
axs[1, 1].plot(temp_steps, inpc, 'tab:red')
axs[1, 1].plot(temp_steps, predict_incp, 'tab:blue')
axs[1, 1].set_title('INPC (2001-2020)')

for ax in axs.flat:
    ax.set(xlabel='tiempo(quincenas)', ylabel='inflacion')

plt.show()

actual, pred = np.array(inpc), np.array(predict_incp)
mae = np.mean(np.abs(((actual - pred) + 0.000001) / (actual + 0.000001)))*100
print("MAE inpc: ")
print(mae)
actual, pred = np.array(subyacente), np.array(predict_sub)
mae = np.mean(np.abs(((actual - pred) + 0.000001) / (actual + 0.000001)))*100
print("MAE subya: ")
print(mae)

model.save('banorte_infla.h5')


