############################
# Import necessary libraries
############################
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from utils.mobility_utils import generate_mobility
from utils.map_utils import recompute_regions
from utils.radio_utils import compute_sinr_dl
from scipy.ndimage import uniform_filter1d

from stable_baselines3 import PPO


############################
# Global variables
############################
MAP_LIMIT = 1000
ALPHA_LOSS = 4
PMACRO = 40
PFEMTO = 0.1
MACRO_BW = 20e6
FEMTO_BW = 1e9
NOISE = 2.5e-14
b = 1 * np.sqrt(2 / np.pi)
SMA_WINDOW = 10

SIM_TIME = 300
TIME_STEP = 0.5
N_USERS = 5
RADIOS_PER_USER = 2

sim_times = np.arange(0, SIM_TIME + TIME_STEP, TIME_STEP)


############################
# Load PPO model
############################
model = PPO.load("trained_agents\proportional_fairness\ppo_cellular_multi_connectivity")


############################
# Load base stations
############################
mat = scipy.io.loadmat("nice_setup_Proteus.mat")
base_stations = mat["BaseStations"]
n_points = base_stations.shape[0]

N_MACRO = 3
N_FEMTO = 10

colors = np.random.rand(n_points, 3)
regions = recompute_regions(n_points, MAP_LIMIT, base_stations, ALPHA_LOSS)


############################
# Generate mobility
############################
sim_input = {
    'V_POSITION_X_INTERVAL': [0, MAP_LIMIT],
    'V_POSITION_Y_INTERVAL': [0, MAP_LIMIT],
    'V_SPEED_INTERVAL': [1, 10],
    'V_PAUSE_INTERVAL': [0, 3],
    'V_WALK_INTERVAL': [30.0, 60.0],
    'V_DIRECTION_INTERVAL': [-180, 180],
    'SIMULATION_TIME': SIM_TIME,
    'NB_NODES': N_USERS
}
s_mobility = generate_mobility(sim_input)

user_positions = []
for node in s_mobility["VS_NODE"]:
    x = np.interp(sim_times, node["V_TIME"], node["V_POSITION_X"])
    y = np.interp(sim_times, node["V_TIME"], node["V_POSITION_Y"])
    user_positions.append((x, y))


############################
# Plots setup
############################
plt.ion()
fig, (ax_map, ax_usage, ax_throughput) = plt.subplots(1, 3, figsize=(18, 6))

# Map with base stations
ax_map.set_xlim(0, MAP_LIMIT)
ax_map.set_ylim(0, MAP_LIMIT)
for j, (x, y, _) in enumerate(base_stations):
    ax_map.plot(x, y, 'o', color=colors[j], markersize=10)
    ax_map.text(x, y + 15, f"P{j+1}", ha='center', fontsize=8)
patches = []
for j, region in enumerate(regions):
    if isinstance(region, Polygon) and not region.is_empty:
        coords = np.array(region.exterior.coords)
        if len(coords) >= 4:
            patch = MplPolygon(coords, closed=True,
                               facecolor=colors[j],
                               edgecolor='none',
                               alpha=0.35)
            patches.append(patch)
ax_map.add_collection(PatchCollection(patches, match_original=True))

# Define user dots and lines
user_dots, assoc_lines = [], []
for u in range(N_USERS):
    dot, = ax_map.plot([], [], '+', color='blue', markersize=10, markeredgewidth=2)
    user_dots.append(dot)
    user_lines = []
    for _ in range(RADIOS_PER_USER):
        line, = ax_map.plot([], [], '-', linewidth=0.8)
        user_lines.append(line)
    assoc_lines.append(user_lines)

# Number of small cells under use
ax_usage.set_xlim(0, SIM_TIME)
ax_usage.set_ylim(0, N_FEMTO + 1)
ax_usage.set_title("Number of small cells under use")
ax_usage.set_xlabel("Time [s]")
ax_usage.set_ylabel("Small cells")
max_cells_line, = ax_usage.plot([0], [N_FEMTO], 'r', label='Total Small cells')
femto_usage_line, = ax_usage.plot([], [], 'g', label='Small cells being used')
usage_text = ax_usage.text(0, N_FEMTO - 1, "Phantom Cells ON: 0", fontsize=9)
ax_usage.legend()

# Throughput plot
ax_throughput.set_xlim(0, SIM_TIME)
ax_throughput.set_ylim(0, 3000)
ax_throughput.set_title("Throughput acumulado")
ax_throughput.set_xlabel("Tiempo [s]")
ax_throughput.set_ylabel("Mbps")
th_plot, = ax_throughput.plot([], [], 'b')


############################
# Simulation loop
############################
live_femto_usage = []
live_throughput = []
live_times = []

bs_user_count = np.zeros(n_points, dtype=int)
for i, t in enumerate(sim_times):
    active_cells = np.zeros(n_points, dtype=bool)
    throughput = 0

    # Build observation for DRL
    user_pos_this_step = []
    for u, (ux, uy) in enumerate(user_positions[:N_USERS]):
        x = ux[i]
        y = uy[i]
        user_pos_this_step.extend([x, y])
        user_dots[u].set_data(x, y)

    obs_vec = np.array(user_pos_this_step + list(bs_user_count), dtype=np.float32).reshape(1, -1)

    # Obtain action from DRL model
    action, _ = model.predict(obs_vec, deterministic=True)
    associations = np.array(action).reshape((N_USERS, RADIOS_PER_USER))

    # Reset BS counter
    bs_user_count = np.zeros(n_points, dtype=int)
    user_bs_sets = [set() for _ in range(n_points)]

    for u in range(N_USERS):
        x = user_positions[u][0][i]
        y = user_positions[u][1][i]

        for r in range(RADIOS_PER_USER):
            bs_idx = associations[u, r]
            if u not in user_bs_sets[bs_idx]:
                bs_user_count[bs_idx] += 1
                user_bs_sets[bs_idx].add(u)
            active_cells[bs_idx] = True

            bs_x, bs_y = base_stations[bs_idx][:2]
            assoc_lines[u][r].set_data([x, bs_x], [y, bs_y])
            assoc_lines[u][r].set_color(colors[bs_idx])

            sinr = compute_sinr_dl(
                (x, y),
                base_stations,
                bs_idx,
                ALPHA_LOSS,
                PMACRO,
                PFEMTO,
                N_MACRO,
                NOISE,
                b
            )
            snr_linear = 10 ** (sinr / 10)
            bw = MACRO_BW if bs_idx < N_MACRO else FEMTO_BW
            user_rate = (bw / bs_user_count[bs_idx]) * np.log2(1 + snr_linear)
            throughput += user_rate

    # Update metrics and live data
    femto_on = np.sum(active_cells[N_MACRO:N_MACRO + N_FEMTO])
    live_femto_usage.append(femto_on)
    live_times.append(t)
    live_throughput.append(throughput / 1e6)

    # Update plot lines
    max_cells_line.set_data([0, t], [N_FEMTO, N_FEMTO])
    femto_usage_line.set_data(live_times, live_femto_usage)
    usage_text.set_text(f"Phantom Cells ON: {femto_on}")
    usage_text.set_position((t, N_FEMTO - 1))

    if len(live_throughput) >= SMA_WINDOW:
        smooth_tp = uniform_filter1d(live_throughput, size=SMA_WINDOW)
        th_plot.set_data(live_times, smooth_tp)

    ax_map.set_title(f"t = {t:.1f} s")
    plt.pause(0.01)
    time.sleep(0.005)

plt.ioff()
plt.show()
