import numpy as np

def generate_mobility(s_input):
    """
    Generates mobility traces for a set of nodes using a Random Waypoint model with bounded movement.

    Parameters
    ----------
    s_input : dict
        Dictionary with the following keys:
            - V_POSITION_X_INTERVAL : [float, float]
            - V_POSITION_Y_INTERVAL : [float, float]
            - V_SPEED_INTERVAL : [float, float]
            - V_PAUSE_INTERVAL : [float, float]
            - V_WALK_INTERVAL : [float, float]
            - V_DIRECTION_INTERVAL : [float, float]
            - SIMULATION_TIME : float
            - NB_NODES : int

    Returns
    -------
    s_mobility : dict
        Dictionary containing per-node mobility data with keys:
            - NB_NODES : int
            - SIMULATION_TIME : float
            - VS_NODE : list of dicts with:
                - V_TIME
                - V_POSITION_X
                - V_POSITION_Y
                - V_SPEED_X
                - V_SPEED_Y
    """
    np.random.seed(42)  # For reproducibility (optional)

    s_mobility = {
        "NB_NODES": s_input["NB_NODES"],
        "SIMULATION_TIME": s_input["SIMULATION_TIME"],
        "VS_NODE": []
    }

    for _ in range(s_input["NB_NODES"]):
        node = {
            "V_TIME": [],
            "V_POSITION_X": [],
            "V_POSITION_Y": [],
            "V_DIRECTION": [],
            "V_SPEED_MAGNITUDE": [],
            "V_IS_MOVING": [],
            "V_DURATION": [],
        }

        x = np.random.uniform(*s_input["V_POSITION_X_INTERVAL"])
        y = np.random.uniform(*s_input["V_POSITION_Y_INTERVAL"])
        t = 0.0
        d = 0.0

        append_point(node, x, y, 0.0, 0.0, False, 0.0, t + d)

        while node["V_TIME"][-1] < s_input["SIMULATION_TIME"]:
            if not node["V_IS_MOVING"][-1]:
                x, y = node["V_POSITION_X"][-1], node["V_POSITION_Y"][-1]
                t = node["V_TIME"][-1]
                d = node["V_DURATION"][-1]
                walk_random_waypoint(node, x, y, d, t, s_input)
            else:
                direction = node["V_DIRECTION"][-1]
                speed = node["V_SPEED_MAGNITUDE"][-1]
                x, y = node["V_POSITION_X"][-1], node["V_POSITION_Y"][-1]
                t = node["V_TIME"][-1]
                d = node["V_DURATION"][-1]
                distance = d * speed
                t_end = t + d
                x_end = x + distance * np.cos(np.radians(direction))
                y_end = y + distance * np.sin(np.radians(direction))
                pause_duration = np.random.uniform(*s_input["V_PAUSE_INTERVAL"])
                d_pause = adjust_duration(t_end, pause_duration, s_input)
                append_point(node, x_end, y_end, 0.0, 0.0, False, d_pause, t_end)

        # Convert scalars to velocity vectors
        vx, vy = [], []
        for speed, angle in zip(node["V_SPEED_MAGNITUDE"], node["V_DIRECTION"]):
            vx.append(speed * np.cos(np.radians(angle)))
            vy.append(speed * np.sin(np.radians(angle)))
        node["V_SPEED_X"] = vx
        node["V_SPEED_Y"] = vy

        # Clean up:
        trim_null_pauses(node)
        trim_final_microstep(node, s_input["SIMULATION_TIME"])

        s_mobility["VS_NODE"].append({
            "V_TIME": node["V_TIME"],
            "V_POSITION_X": node["V_POSITION_X"],
            "V_POSITION_Y": node["V_POSITION_Y"],
            "V_SPEED_X": node["V_SPEED_X"],
            "V_SPEED_Y": node["V_SPEED_Y"]
        })

    return s_mobility


def append_point(node, x, y, direction, speed, is_moving, duration, time):
    node["V_POSITION_X"].append(x)
    node["V_POSITION_Y"].append(y)
    node["V_DIRECTION"].append(direction)
    node["V_SPEED_MAGNITUDE"].append(speed)
    node["V_IS_MOVING"].append(is_moving)
    node["V_DURATION"].append(duration)
    node["V_TIME"].append(time)


def walk_random_waypoint(node, x, y, d, t, s_input):
    duration = adjust_duration(t + d, np.random.uniform(*s_input["V_WALK_INTERVAL"]), s_input)
    direction = np.random.uniform(*s_input["V_DIRECTION_INTERVAL"])
    speed = np.random.uniform(*s_input["V_SPEED_INTERVAL"])
    distance = duration * speed

    if distance == 0:
        append_point(node, x, y, direction, speed, True, duration, t + d)
        return

    finished = False
    while not finished:
        x_next = x + distance * np.cos(np.radians(direction))
        y_next = y + distance * np.sin(np.radians(direction))
        new_dir = direction
        flag_outside = False

        if x_next > s_input["V_POSITION_X_INTERVAL"][1]:
            flag_outside = True
            new_dir = 180 - direction
            x_next = s_input["V_POSITION_X_INTERVAL"][1]
            y_next = y + (x_next - x) * np.tan(np.radians(direction))
        elif x_next < s_input["V_POSITION_X_INTERVAL"][0]:
            flag_outside = True
            new_dir = 180 - direction
            x_next = s_input["V_POSITION_X_INTERVAL"][0]
            y_next = y + (x_next - x) * np.tan(np.radians(direction))

        if y_next > s_input["V_POSITION_Y_INTERVAL"][1]:
            flag_outside = True
            new_dir = -direction
            y_next = s_input["V_POSITION_Y_INTERVAL"][1]
            x_next = x + (y_next - y) / np.tan(np.radians(direction))
        elif y_next < s_input["V_POSITION_Y_INTERVAL"][0]:
            flag_outside = True
            new_dir = -direction
            y_next = s_input["V_POSITION_Y_INTERVAL"][0]
            x_next = x + (y_next - y) / np.tan(np.radians(direction))

        step_distance = np.hypot(x_next - x, y_next - y)
        step_duration = adjust_duration(t + d, step_distance / speed, s_input)

        append_point(node, x, y, direction, speed, True, step_duration, t + d)

        if flag_outside:
            d += step_duration
            distance -= step_distance
            x, y = x_next, y_next
            direction = new_dir
        else:
            finished = True


def adjust_duration(current_time, duration, s_input):
    """
    Shortens the duration if the time would exceed the simulation limit.
    """
    if current_time + duration >= s_input["SIMULATION_TIME"]:
        return s_input["SIMULATION_TIME"] - current_time
    return duration


def trim_null_pauses(node):
    """
    Remove points with 0 duration except the last one.
    """
    if len(node["V_DURATION"]) < 2:
        return
    mask = np.array(node["V_DURATION"][:-1]) > 0
    for key in ["V_TIME", "V_POSITION_X", "V_POSITION_Y", "V_DIRECTION",
                "V_SPEED_MAGNITUDE", "V_IS_MOVING", "V_DURATION"]:
        node[key] = list(np.array(node[key])[np.append(mask, True)])


def trim_final_microstep(node, sim_time):
    """
    Ensures the last time step is exactly SIMULATION_TIME and clean.
    """
    if len(node["V_TIME"]) >= 2 and abs(node["V_TIME"][-1] - node["V_TIME"][-2]) < 1e-14:
        for key in ["V_TIME", "V_POSITION_X", "V_POSITION_Y", "V_DIRECTION",
                    "V_SPEED_MAGNITUDE", "V_IS_MOVING", "V_DURATION",
                    "V_SPEED_X", "V_SPEED_Y"]:
            node[key].pop()

    node["V_TIME"][-1] = sim_time
    node["V_DURATION"][-1] = 0
    node["V_SPEED_MAGNITUDE"][-1] = 0
    node["V_SPEED_X"][-1] = 0
    node["V_SPEED_Y"][-1] = 0
