import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
model = tf.keras.models.load_model('banorte_infla.h5')
print(model)

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


temp_steps = []
cont = 0
for i in range(15):
    temp_steps.append(cont)
    cont += 1


merc_data = merc_data[-1]
merc_data = np.reshape(merc_data, (1,24))
serv_data = serv_data[-1]
serv_data = np.reshape(serv_data, (1,24))
agro_data = agro_data[-1]
agro_data = np.reshape(agro_data, (1,24))
energ_data = energ_data[-1]
energ_data = np.reshape(energ_data, (1,24))
sub_data = sub_data[-1]
sub_data = np.reshape(sub_data, (1,24))
not_data = not_data[-1]
not_data = np.reshape(not_data, (1,24))
inpc_data = inpc_data[-1]
inpc_data = np.reshape(inpc_data, (1,24))

prediccion_incp = []
prediccion_suby = []
prediccion_notsub = []

dt2 = pd.read_csv('pasado2021.csv')
inpc_real = dt2["INPC"]
subya_real = dt2["Subyacente"]
not_real = dt2["Nosubyacente"]
inpc_real = np.array(inpc_real)
subya_real = np.array(subya_real)
not_real  = np.array(not_real)

for i in range(15):
    p_mercancias, p_servicios, p_agropecuarios, p_energia, predict_sub, predict_not, predict_incp = model.predict(
        [merc_data, serv_data, agro_data, energ_data, sub_data, not_data, inpc_data])
    merc_data = np.delete(merc_data, 0)
    merc_data = np.append(merc_data, p_mercancias).reshape((1,24))
    serv_data = np.delete(serv_data, 0)
    serv_data = np.append(serv_data, p_servicios).reshape((1, 24))
    agro_data = np.delete(agro_data, 0)
    agro_data = np.append(agro_data, p_agropecuarios).reshape((1, 24))
    energ_data = np.delete(energ_data, 0)
    energ_data = np.append(energ_data, p_energia).reshape((1, 24))
    sub_data = np.delete(sub_data, 0)
    sub_data = np.append(sub_data, predict_sub).reshape((1, 24))
    not_data = np.delete(not_data, 0)
    not_data = np.append(not_data, predict_not).reshape((1, 24))
    inpc_data = np.delete(inpc_data, 0)
    inpc_data = np.append(inpc_data, predict_incp).reshape((1, 24))
    prediccion_incp.append(predict_incp)
    prediccion_suby.append(predict_sub)
    prediccion_notsub.append(predict_not)

prediccion_incp = np.reshape(prediccion_incp, (15))
prediccion_suby = np.reshape(prediccion_suby, (15))
prediccion_notsub = np.reshape(prediccion_notsub, (15))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(temp_steps, inpc_real, 'tab:red')
axs[0, 0].plot(temp_steps, prediccion_incp, 'tab:blue')
axs[0, 0].set_title('INPC (2021)')
axs[0, 1].plot(temp_steps, subya_real, 'tab:red')
axs[0, 1].plot(temp_steps, prediccion_suby, 'tab:blue')
axs[0, 1].set_title('Subyacente (2021)')
axs[1, 0].plot(temp_steps, not_real, 'tab:red')
axs[1, 0].plot(temp_steps, prediccion_notsub, 'tab:blue')
axs[1, 0].set_title('No Subyacente (2021)')
axs[1, 1].plot(temp_steps, inpc_real, 'tab:red')
axs[1, 1].plot(temp_steps, prediccion_incp, 'tab:blue')
axs[1, 1].set_title('INPC (2021)')
print("INCP real:")
print(inpc_real)
print("INCP programa")
print(prediccion_incp)
print("Subya real:")
print(subya_real)
print("Subya programa")
print(prediccion_suby)
plt.show()

actual, pred = np.array(inpc_real), np.array(prediccion_incp)
mae = np.mean(np.abs((actual - pred) / actual))*100
print("MAE INCP: ")
print(mae)
actual, pred = np.array(subya_real), np.array(prediccion_suby)
mae = np.mean(np.abs((actual - pred) / actual))*100
print("MAE Subya: ")
print(mae)
