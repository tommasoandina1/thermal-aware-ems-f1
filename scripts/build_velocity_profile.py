import numpy as np
import fastf1
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os

os.makedirs('./f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('./f1_cache')

# load a session and its telemetry data of qualifyng
session = fastf1.get_session(2026, 'Canada Grand Prix', 'Q')
session.load()

#Select the lap that we want to compare
lap = session.laps.pick_fastest()
tel = lap.get_telemetry()


time = tel['Time'].dt.total_seconds().values # s
dt = 0.1
N = int(time[-1]/dt)
t = np.arange(0, time[-1], dt)
vel = tel['Speed'].values / 3.6 # km/h -> m/s
v = np.interp(t, time, vel)
a = savgol_filter(v, window_length=12, polyorder=3, deriv=1, delta=dt)

np.save('/app/data/qualifying_Canada/Canada_qualifying.npy', np.stack([t, v, a]))

plt.figure(figsize=(12, 10), facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.plot(t,v)
plt.ylabel('Velocity (m/s)',fontsize = 12)
plt.xlabel('time (s)',fontsize = 12)

plt.subplot(2,1,2)
plt.plot(t,a)
plt.ylabel('Acceleration (m/s^2)',fontsize = 12)
plt.xlabel('time (s)',fontsize = 12)

plt.tight_layout()
plt.savefig('/app/img/qualifying_profile.pdf')

#MULTI LAPS
# load a session and its telemetry data of race
session_race = fastf1.get_session(2026, 'Canada Grand Prix', 'R')
session_race.load()
laps_5 = session_race.laps.pick_drivers('ANT').pick_laps([3,4,5,6,7])

telemetry_5 = laps_5.get_telemetry()
time_raw = telemetry_5['Time'].dt.total_seconds().values
time_continuous = time_raw - time_raw[0]

dt = 0.1
t_5_lap = np.arange(0, time_continuous[-1], dt)

vel_5 = telemetry_5['Speed'].values / 3.6
vel_5_lap = np.interp(t_5_lap, time_continuous, vel_5)

a_5_lap = savgol_filter(vel_5_lap, window_length=12, polyorder=3, deriv=1, delta=dt)

lap_interp = np.zeros_like(t_5_lap)
t_actual = 0.0

for _, lap in laps_5.iterlaps():
    start_s = lap['LapTime'].total_seconds()
    end_s = t_actual+start_s
    mask = (t_5_lap >= t_actual) & (t_5_lap <= end_s)
    lap_interp[mask] = lap['LapNumber']
    t_actual=end_s

lap_interp[lap_interp == 0] = laps_5.iloc[-1]['LapNumber']


np.save('/app/data/multi_lap_Canada/Canada_5laps.npy', np.stack([t_5_lap,lap_interp, vel_5_lap, a_5_lap]))

#Plot 5 lap
plt.figure(figsize=(22, 12), facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.plot(t_5_lap, vel_5_lap)
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')
plt.title('Velocity Profile - 5 consecutive laps ')


plt.subplot(2,1,2)
plt.plot(t_5_lap, lap_interp, color='orange')
plt.ylabel('Lap Number')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('/app/img/multi_lap.pdf', dpi=300, bbox_inches='tight')