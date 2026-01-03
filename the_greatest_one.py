
#写字高度，在这个高度以下才记录轨迹点
write_limit=0.1
#笔画轨迹移动时间
move_time=0.5
#提笔轨迹移动时间
hang_time=0.5
# 轨迹点最小间距，避免冗余
min_dist=0.001
#运行总时长
sim_time=1000



import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import scipy as sp

xml_path = '../../models/universal_robots_ur5e/scene.xml'
simend = sim_time
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# UR5e关节限位（最小值，最大值），6关节对应
UR5E_JOINT_LIMITS = np.array([
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi]
])

# =========================
# 控制参数（新增，不改你已有变量名）
# =========================
DAMP = 1e-2          # 阻尼项：防奇异
POS_GAIN = 1.2       # 位置跟踪增益
ROT_GAIN = 0.6       # 姿态跟踪增益（笔尖朝下）
DQ_LIMIT = 0.12      # 每步关节增量限幅（比你原来0.1略大一点，能更快落笔）
EE_SITE_ID = 0       # 仍用你原来的site=0（不改变量名；若site不是笔尖请自行改成正确id）

# Helper function
def IK_controller(model, data, X_ref, q_pos):
    """
    最小化改动版：
    - 由 3D位置IK 升级为 6D（位置+姿态）阻尼最小二乘IK
    - 保持函数名、入参、返回不变
    """
    position_Q = data.site_xpos[EE_SITE_ID].copy()
    site_xmat = data.site_xmat[EE_SITE_ID].reshape(3, 3).copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, EE_SITE_ID)

    # 位置误差
    dX_pos = (X_ref - position_Q) * POS_GAIN

    # 姿态误差：让末端z轴对齐世界 -Z（笔尖朝下）
    z_world_des = np.array([0.0, 0.0, -1.0])
    z_site = site_xmat[:, 2]
    dX_rot = np.cross(z_site, z_world_des) * ROT_GAIN

    # 6D任务
    J = np.vstack((jacp, jacr))
    dX = np.hstack((dX_pos, dX_rot))

    # 阻尼最小二乘：dq = J^T (J J^T + λ^2 I)^-1 dX
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (DAMP**2) * np.eye(6), dX)

    dq = np.clip(dq, -DQ_LIMIT, DQ_LIMIT)
    q_new = q_pos + dq

    # 关节角度限位检查
    for i in range(len(q_new)):
        q_new[i] = np.clip(q_new[i], UR5E_JOINT_LIMITS[i, 0], UR5E_JOINT_LIMITS[i, 1])

    return q_new

def init_controller(model,data):
    pass

def controller(model, data):
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05*yoffset, scene, cam)

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Init GLFW
glfw.init()
window = glfw.create_window(1920, 1080, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 179.83
cam.elevation = 87.1633
cam.distance = 2.22
cam.lookat = np.array([0.29723477517870245, 0.28277006411151073, 0.6082647377843177])

init_controller(model,data)
mj.set_mjcb_control(controller)

# =========================
# 改动：更“写字友好”的初始位姿（只改数值，不改变量名 init_qpos）
# 目的：远离腕部奇异、让笔尖朝下更自然
# =========================
init_qpos = np.array([-1.57, -1.20, 2.05, -2.45, -1.57, 0.0])
data.qpos[:] = init_qpos
cur_q_pos = init_qpos.copy()
mj.mj_forward(model, data)

# Trajectory visualization
traj_points = []
MAX_TRAJ = 5000
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])
MIN_DISTANCE = min_dist
WRITE_Z_THRESHOLD = write_limit

######################################
class TrajectorySegment:
    def __init__(self, start_point, end_point, control_point=None, interp_type='linear', duration=3.0):
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.control_point = np.array(control_point) if control_point is not None else None
        self.interp_type = interp_type
        self.duration = duration
        self.completed = False

# ===== 你的轨迹段：原样保留 =====
trajectory_segments = [
    TrajectorySegment(start_point=[0.19, 0.5, 0.3], end_point=[-0.19, 0.5, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.19, 0.5, 0.1], end_point=[-0.08, 0.5, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.08, 0.5, 0.1], end_point=[-0.13, 0.52, 0.1], control_point=[-0.08, 0.5, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.13, 0.52, 0.1], end_point=[-0.13, 0.47, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.13, 0.47, 0.1], end_point=[-0.13, 0.5, 0.1], control_point=[-0.13, 0.47, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.13, 0.5, 0.1], end_point=[-0.19, 0.47, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.19, 0.47, 0.1], end_point=[-0.13, 0.5, 0.1], control_point=[-0.19, 0.47, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.13, 0.5, 0.1], end_point=[-0.07, 0.47, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.07, 0.47, 0.1], end_point=[-0.17, 0.46, 0.1], control_point=[-0.07, 0.47, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.17, 0.46, 0.1], end_point=[-0.10, 0.46, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.10, 0.46, 0.1], end_point=[-0.10, 0.46, 0.1], control_point=[-0.10, 0.46, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.10, 0.46, 0.1], end_point=[-0.13, 0.45, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.13, 0.45, 0.1], end_point=[-0.13, 0.45, 0.1], control_point=[-0.13, 0.45, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.13, 0.45, 0.1], end_point=[-0.13, 0.41, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.13, 0.41, 0.1], end_point=[-0.13, 0.41, 0.1], control_point=[-0.13, 0.41, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.13, 0.41, 0.1], end_point=[-0.15, 0.41, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.15, 0.41, 0.1], end_point=[-0.19, 0.44, 0.1], control_point=[-0.15, 0.41, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.19, 0.44, 0.1], end_point=[-0.08, 0.44, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.08, 0.44, 0.1], end_point=[-0.01, 0.52, 0.1], control_point=[-0.08, 0.44, 0.3], interp_type='bezier', duration=2.0),

    TrajectorySegment(start_point=[-0.01, 0.52, 0.1], end_point=[0.01, 0.5, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.01, 0.5, 0.1], end_point=[-0.055, 0.49, 0.1], control_point=[0.01, 0.5, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.055, 0.49, 0.1], end_point=[0.055, 0.49, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.055, 0.49, 0.1], end_point=[0.025, 0.49, 0.1], control_point=[0.055, 0.49, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.025, 0.49, 0.1], end_point=[-0.05, 0.4, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[-0.05, 0.4, 0.1], end_point=[-0.025, 0.49, 0.1], control_point=[-0.05, 0.4, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[-0.025, 0.49, 0.1], end_point=[0.05, 0.4, 0.1], interp_type='linear', duration=1.0),

    TrajectorySegment(start_point=[0.05, 0.4, 0.1], end_point=[0.095, 0.52, 0.1], control_point=[0.05, 0.4, 0.3], interp_type='bezier', duration=2.0),

    TrajectorySegment(start_point=[0.095, 0.52, 0.1], end_point=[0.095, 0.4, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.095, 0.4, 0.1], end_point=[0.08, 0.485, 0.1], control_point=[0.095, 0.4, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.08, 0.485, 0.1], end_point=[0.075, 0.465, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.075, 0.465, 0.1], end_point=[0.105, 0.485, 0.1], control_point=[0.075, 0.465, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.105, 0.485, 0.1], end_point=[0.112, 0.48, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.112, 0.48, 0.1], end_point=[0.115, 0.51, 0.1], control_point=[0.112, 0.48, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.115, 0.51, 0.1], end_point=[0.185, 0.51, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.185, 0.51, 0.1], end_point=[0.125, 0.475, 0.1], control_point=[0.185, 0.51, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.125, 0.475, 0.1], end_point=[0.125, 0.44, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.125, 0.44, 0.1], end_point=[0.175, 0.475, 0.1], control_point=[0.125, 0.44, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.175, 0.475, 0.1], end_point=[0.175, 0.44, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.175, 0.44, 0.1], end_point=[0.125, 0.475, 0.1], control_point=[0.175, 0.44, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.125, 0.475, 0.1], end_point=[0.175, 0.475, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.175, 0.475, 0.1], end_point=[0.125, 0.4575, 0.1], control_point=[0.175, 0.475, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.125, 0.4575, 0.1], end_point=[0.175, 0.4575, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.175, 0.4575, 0.1], end_point=[0.125, 0.44, 0.1], control_point=[0.175, 0.4575, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.125, 0.44, 0.1], end_point=[0.175, 0.44, 0.1], interp_type='linear', duration=1.0),
    TrajectorySegment(start_point=[0.175, 0.44, 0.1], end_point=[0.115, 0.415, 0.1], control_point=[0.175, 0.44, 0.3], interp_type='bezier', duration=1.5),
    TrajectorySegment(start_point=[0.115, 0.415, 0.1], end_point=[0.185, 0.415, 0.1], interp_type='linear', duration=1.0),

    # “2”
    # ===== 恒 -> 2：抬笔-平移-落笔（三段linear，duration变为原来的2倍）=====
TrajectorySegment(
    start_point=[0.185, 0.415, 0.1],      # 恒字最后一笔终点（你第44段的end_point）
    end_point=[0.185, 0.415, 0.3],        # 抬笔到0.3
    interp_type='linear',
    duration=hang_time * 2.0              # 原来hang_time，改为2倍
),
TrajectorySegment(
    start_point=[0.185, 0.415, 0.3],
    end_point=[-0.331, 0.205, 0.3],       # 平移到2起点正上方
    interp_type='linear',
    duration=move_time * 2.0              # 原来move_time，改为2倍
),
TrajectorySegment(
    start_point=[-0.331, 0.205, 0.3],
    end_point=[-0.331, 0.205, 0.1],       # 落笔到2起点
    interp_type='linear',
    duration=hang_time * 2.0              # 原来hang_time，改为2倍
),
    TrajectorySegment(start_point=[-0.331, 0.205, 0.1], end_point=[-0.29, 0.181, 0.1], control_point=[-0.3, 0.22, 0.1], interp_type='bezier', duration=2.0),
    TrajectorySegment(start_point=[-0.29, 0.181, 0.1], end_point=[-0.29, 0.181, 0.1], control_point=[-0.29, 0.181, 0.3], interp_type='bezier', duration=2.0),
    TrajectorySegment(start_point=[-0.29, 0.181, 0.1], end_point=[-0.33, 0.14, 0.1], interp_type='linear', duration=2.0),
]

current_segment_index = 0
segment_start_time = 0.0
X_ref = trajectory_segments[0].start_point

def LinearInterpolate(q0, q1, t, t_total):
    u = t / t_total
    u = np.clip(u, 0, 1)
    u = np.sin(u * np.pi / 2)
    return q0 + u * (q1 - q0)

def QuadBezierInterpolate(q0, q1, q2, t, t_total):
    u = t / t_total
    u = np.clip(u, 0, 1)
    u = np.sin(u * np.pi / 2)
    return (1 - u)**2 * q0 + 2 * u * (1 - u) * q1 + u**2 * q2

def get_reference_position(current_time):
    global current_segment_index, segment_start_time, X_ref, trajectory_segments

    if current_segment_index >= len(trajectory_segments):
        return trajectory_segments[-1].end_point

    current_segment = trajectory_segments[current_segment_index]
    segment_elapsed_time = current_time - segment_start_time

    if segment_elapsed_time >= current_segment.duration:
        current_segment.completed = True
        X_ref = current_segment.end_point
        current_segment_index += 1
        if current_segment_index < len(trajectory_segments):
            segment_start_time = current_time
            current_segment = trajectory_segments[current_segment_index]
            X_ref = current_segment.start_point

    if current_segment.interp_type == 'linear':
        X_ref = LinearInterpolate(current_segment.start_point, current_segment.end_point,
                                  segment_elapsed_time, current_segment.duration)
    elif current_segment.interp_type == 'bezier' and current_segment.control_point is not None:
        X_ref = QuadBezierInterpolate(current_segment.start_point, current_segment.control_point,
                                      current_segment.end_point, segment_elapsed_time, current_segment.duration)
    return X_ref

# ===== 新增：起始阶段不记录，避免笔尖初始高度偏低造成墨迹 =====
NO_RECORD_TIME = 0.6

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj_end_eff_pos = data.site_xpos[EE_SITE_ID]

        # Store trajectory: z<=阈值 且 超过启动保护时间才记录
        if data.time >= NO_RECORD_TIME and mj_end_eff_pos[2] <= WRITE_Z_THRESHOLD:
            if len(traj_points) == 0:
                traj_points.append(mj_end_eff_pos.copy())
            else:
                dist = np.linalg.norm(mj_end_eff_pos - traj_points[-1])
                if dist >= MIN_DISTANCE:
                    traj_points.append(mj_end_eff_pos.copy())
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)

        cur_q_pos = data.qpos.copy()
        X_ref = get_reference_position(data.time)
        cur_ctrl = IK_controller(model, data, X_ref, cur_q_pos)
        data.ctrl[:] = cur_ctrl

        mj.mj_step(model, data)

    if data.time >= simend:
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # draw traj
    for p in traj_points:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.002, 0.002, 0.002])
        geom.pos[:] = p
        geom.mat[:] = np.eye(3)
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0

    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
