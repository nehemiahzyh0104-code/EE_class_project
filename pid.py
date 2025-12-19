import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import time

xml_path = '../../models/Pendulum/pendulum.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.normpath(os.path.join(dirname, xml_path))

# --- PID controller ---
def pid(kp, ki, kd, state, dt):
    """
    Args:
        kp, ki, kd: Controller gains
        state: Dictionary containing 'error', 'integral', and 'last_error'
        dt: Time step for discrete integration/differentiation
    Returns:
        output: Control torque
    """
    # for P
    current_error = state['error']
    # for I
    state['integral'] += current_error * dt
    # for D
    derivative = (current_error - state['last_error']) / dt
    # PID output
    output = (kp * current_error) + (ki * state['integral']) + (kd * derivative)
    
    # update state
    state['last_error'] = current_error
    
    return output

model = mujoco.MjModel.from_xml_path(abspath)
data = mujoco.MjData(model)

# control parameters
# TODO: tune these gains
KP = 97.25
KI = 4
KD = 12
TARGET_ANGLE = np.pi  
DT = model.opt.timestep

# initialize control state
control_state = {
    'error': 0.0,
    'integral': 0.0,
    'last_error': 0.0
}

# --- data storage for plotting ---
time_history = []
error_history = []

# print model path
print(f"Loading model from: {abspath}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_sim_time = data.time
    
    while viewer.is_running() and data.time - start_sim_time < 10.0:
        step_start = time.time()
        control_state['error'] = TARGET_ANGLE - data.qpos[0]
        torque = pid(KP, KI, KD, control_state, DT)
        data.ctrl[0] = torque
        
        time_history.append(data.time)
        error_history.append(control_state['error'])

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

# --- plot ---
plt.figure(figsize=(10, 6))
plt.plot(time_history, error_history, label='Error (target - current)', color='forestgreen')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title('PID Control Performance: Error Convergence')
plt.xlabel('Time (s)')
plt.ylabel('Error (rad)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()