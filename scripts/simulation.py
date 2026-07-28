import numpy as np

from plant.vehicle_dynamics import aero_drag, rolling_resistance, longitudinal_dynamics
from plant.battery import battery_step, coef_cubic
from plant.powertrain import powertrain
from plant.parameters import params
from controller.rule_based_controller import rule_based_split
import matplotlib.pyplot as plt


data_quali = np.load('data/qualifying_Canada/Canada_qualifying.npy')

t = data_quali[0,:]
v = data_quali[1,:]
a = data_quali[2,:]



#Dynamics of the car for quali
F_aero = np.array(aero_drag(v,params))
F_rolling = rolling_resistance(params) * np.ones(len(v))
F_x  = np.zeros(len(v))
Pm = np.zeros(len(v))
P_gb = np.zeros(len(v))


for k in range(len(v)):
    F_x[k],Pm[k],P_gb[k] = longitudinal_dynamics(v[k],a[k],params)

np.save('data/qualifying_Canada/power_domand_quali.npy',np.stack([Pm, P_gb]))

## 5 laps
data_5_laps = np.load('data/multi_lap_Canada/Canada_5laps.npy')

t_5_lap= data_5_laps[0,:]
lap_interp= data_5_laps[1,:]
vel_5_lap=  data_5_laps[2,:]
a_5_lap =  data_5_laps[3,:]


#Dynamics of the car for 5 laps
F_aero_5_laps = np.array(aero_drag(vel_5_lap,params))
F_rolling_5_laps = rolling_resistance(params) * np.ones(len(vel_5_lap))
F_x_5_laps  = np.zeros(len(vel_5_lap))
Pm_5_laps = np.zeros(len(vel_5_lap))
P_gb_5_laps = np.zeros(len(vel_5_lap))

for k in range(len(vel_5_lap)):
    F_x_5_laps[k],Pm_5_laps[k],P_gb_5_laps[k] = longitudinal_dynamics(vel_5_lap[k],a_5_lap[k],params)

np.save('data/qualifying_Canada/power_domand_mulilap.npy',np.stack([Pm_5_laps, P_gb_5_laps]))



#Plot 5 lap
plt.figure(figsize=(22, 12), facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.plot(t, P_gb, )
plt.ylabel('Power (W)',fontsize = 16)
plt.xlabel('Time (s)',fontsize = 16)
plt.title('Power Domand at Gearbox for the quali',fontsize = 16)


plt.subplot(2,1,2)
plt.plot(t_5_lap, P_gb_5_laps)
plt.ylabel('Power (W)',fontsize = 16)
plt.xlabel('Time (s)',fontsize = 16)
plt.tight_layout()
plt.title('Power Domand at Gearbox for 5 laps',fontsize = 16)
plt.savefig('/app/img/multi_lap_powerdomand.png', dpi=300, bbox_inches='tight')