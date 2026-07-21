import numpy as np

from plant.vehicle_dynamics import aero_drag, rolling_resistance, longitudinal_dynamics
from plant.battery import battery_step, coef_cubic
from plant.powertrain import powertrain
from plant.parameters import params
from controller.ruled_based import rule_based_split
import matplotlib.pyplot as plot

data = np.load('data/Canada_qualifying.npy')

t = data[0,:]
v = data[1,:]
a = data[2,:]

dt = 0.1 #simulation time
N = int(t[-1]/dt)

#Dynamics of the car
F_aero = np.array(aero_drag(v,params))
F_rolling = rolling_resistance(params) * np.ones(len(v))
F_x  = np.zeros(len(v))
Pm = np.zeros(len(v))
P_gb = np.zeros(len(v))

for k in range(len(v)):
    F_x[k],Pm[k],P_gb[k] = longitudinal_dynamics(v[k],a[k],params)

np.save('data/power_domand.npy',np.stack([Pm, P_gb]))

# #Battery of the car
# SoC = np.zeros(N+1)
# SoC[0] = 0.9

# E_deploy = np.zeros(N+1)
# E_deploy[0] = 0
# E_recharge = np.zeros(N+1)
# E_recharge[0] = 0

# P_ICE_real = np.zeros(N+1)
# P2 = np.zeros(N+1)
# P_mech_MGU_K_real = np.zeros(N+1)
# P2_k = np.zeros(N+1)
# shortfall = np.zeros(N+1)
# u_split = np.zeros(N+1)
# U2 = np.zeros(N+1)
# U2[0] = 300

# I2= np.zeros(N+1)
# for k in range(N):
#     u_split[k] = rule_based_split(SoC[k],P_gb[k],E_deploy[k],params)
#     P_ICE_real[k+1], P2[k+1], P_mech_MGU_K_real[k+1], shortfall[k+1], E_deploy[k+1], E_recharge[k+1] = powertrain(P_gb[k], u_split[k], E_deploy[k], E_recharge[k], params, dt) 
#     SoC[k+1], U2[k+1], I2[k+1] = battery_step(SoC[k], P2[k+1], coef_cubic, params,dt)