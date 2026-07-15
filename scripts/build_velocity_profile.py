import numpy as np
import fastf1
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


fastf1.Cache.enable_cache('./f1_cache')

# load a session and its telemetry data
session = fastf1.get_session(2025, 'Spanish Grand Prix', 'Q')
session.load()

#Select the lap that we want to compare
lap = session.laps.pick_fastest()
tel = lap.get_telemetry()


time = tel['Time'].dt.total_seconds().values # s 
dt = 0.1
N = int(time[-1]/dt)
t = np.arange(0, time[-1], dt)
vel = tel['Speed'].values / 3.6  # km/h -> m/s
v = np.interp(t, time, vel)
a = savgol_filter(v, window_length=15, polyorder=3, deriv=1, delta=dt)

np.save('data/Spanish_qualifying.npy', np.stack([t, v, a]))


fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(t,v)
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('time (s)')

ax2.plot(t,a)
ax2.set_ylabel('Acceleration (m^2/s)')
ax2.set_xlabel('time (s)')

plt.savefig('profilo.png')
