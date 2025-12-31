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

# Helper function
def IK_controller(model, data, X_ref, q_pos):
    # Compute Jacobian（修正关节索引：UR5e为6关节，末端site对应body ID需确认）
    position_Q = data.site_xpos[0]

    jacp = np.zeros((3, model.nv))  # 用model.nv获取关节数量，适配6关节
    jacr = np.zeros((3, model.nv))
    # 计算位置+姿态雅可比（最后一个参数为site ID，修正为0）
    mj.mj_jacSite(model, data, jacp, jacr, 0)

    J = jacp.copy()
    Jinv = np.linalg.pinv(J, rcond=1e-6)  # 增加伪逆条件数，提高稳定性

    # Reference point
    X = position_Q.copy()
    dX = X_ref - X
    # 增加阻尼系数，抑制抖动
    dX = 0.5 * dX

    # Compute control input
    dq = Jinv @ dX

    # 关节角度增量限幅，避免突变
    dq = np.clip(dq, -0.1, 0.1)
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

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

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

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Camera configuration
cam.azimuth =  179.8300000000001
cam.elevation =  87.16333333333334
cam.distance =  2.22
cam.lookat = np.array([ 0.29723477517870245 , 0.28277006411151073 , 0.6082647377843177 ])

# Initialize the controller
init_controller(model,data)

# Set the controller
mj.set_mjcb_control(controller)

# Initialize joint configuration
init_qpos = np.array([-1.6353559, -1.28588984, 2.14838487, -2.61087434, -1.5903009, -0.06818645])
data.qpos[:] = init_qpos
cur_q_pos = init_qpos.copy()

# Trajectory visualization
traj_points = []
MAX_TRAJ = 5000
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])
MIN_DISTANCE = min_dist  # 轨迹点最小间距，避免冗余
WRITE_Z_THRESHOLD = write_limit  # 写字阈值：z≤0.11时写字，z>0.11时不写字

######################################
### 定义多个轨迹段 ###
class TrajectorySegment:
    def __init__(self, start_point, end_point, control_point=None, interp_type='linear', duration=3.0):
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.control_point = np.array(control_point) if control_point is not None else None
        self.interp_type = interp_type
        self.duration = duration
        self.completed = False

# 定义书写路径
trajectory_segments = [
    # ========== 李 字 ==========
    # 1. 从第一笔起始点正上方z=0.3处直线移动到第一笔起始点
    TrajectorySegment(
        start_point=[0.19, 0.5, 0.3],  # 第一笔起始点(F1,G1)的正上方z=0.3处
        end_point=[-0.19, 0.5, 0.1],   # 第一笔起始点(F1,G1), z=0.1
        interp_type='linear',
        duration=1.0
    ),
    
    # 2. 书写第一笔（第一段）
    TrajectorySegment(
        start_point=[-0.19, 0.5, 0.1],   # F1, G1
        end_point=[-0.08, 0.5, 0.1],     # F2, G2
        interp_type='linear',
        duration=1.0
    ),
    
    # 3. 移动到第二笔起始点（控制点为第一笔终止点正上方）
    TrajectorySegment(
        start_point=[-0.08, 0.5, 0.1],     # 第一笔终止点(F2,G2)
        end_point=[-0.13, 0.52, 0.1],      # 第二笔起始点(F3,G3)
        control_point=[-0.08, 0.5, 0.3],   # 第一笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 4. 书写第二笔
    TrajectorySegment(
        start_point=[-0.13, 0.52, 0.1],   # F3, G3
        end_point=[-0.13, 0.47, 0.1],     # F4, G4
        interp_type='linear',
        duration=1.0
    ),
    
    # 5. 移动到第三笔起始点
    TrajectorySegment(
        start_point=[-0.13, 0.47, 0.1],    # 第二笔终止点
        end_point=[-0.13, 0.5, 0.1],       # 第三笔起始点(F5,G5)
        control_point=[-0.13, 0.47, 0.3],  # 第二笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 6. 书写第三笔
    TrajectorySegment(
        start_point=[-0.13, 0.5, 0.1],     # F5, G5
        end_point=[-0.19, 0.47, 0.1],      # F6, G6
        interp_type='linear',
        duration=1.0
    ),
    
    # 7. 移动到第四笔起始点
    TrajectorySegment(
        start_point=[-0.19, 0.47, 0.1],    # 第三笔终止点
        end_point=[-0.13, 0.5, 0.1],       # 第四笔起始点(F7,G7)
        control_point=[-0.19, 0.47, 0.3],  # 第三笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 8. 书写第四笔
    TrajectorySegment(
        start_point=[-0.13, 0.5, 0.1],     # F7, G7
        end_point=[-0.07, 0.47, 0.1],      # F8, G8
        interp_type='linear',
        duration=1.0
    ),
    
    # 9. 移动到第五笔起始点
    TrajectorySegment(
        start_point=[-0.07, 0.47, 0.1],    # 第四笔终止点
        end_point=[-0.17, 0.46, 0.1],      # 第五笔起始点(F9,G9)
        control_point=[-0.07, 0.47, 0.3],  # 第四笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 10. 书写第五笔
    TrajectorySegment(
        start_point=[-0.17, 0.46, 0.1],    # F9, G9
        end_point=[-0.10, 0.46, 0.1],      # F10, G10
        interp_type='linear',
        duration=1.0
    ),
    
    # 11. 移动到第六笔起始点
    TrajectorySegment(
        start_point=[-0.10, 0.46, 0.1],    # 第五笔终止点
        end_point=[-0.10, 0.46, 0.1],      # 第六笔起始点(F11,G11)
        control_point=[-0.10, 0.46, 0.3],  # 第五笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 12. 书写第六笔
    TrajectorySegment(
        start_point=[-0.10, 0.46, 0.1],    # F11, G11
        end_point=[-0.13, 0.45, 0.1],      # F12, G12
        interp_type='linear',
        duration=1.0
    ),
    
    # 13. 移动到第七笔起始点
    TrajectorySegment(
        start_point=[-0.13, 0.45, 0.1],    # 第六笔终止点
        end_point=[-0.13, 0.45, 0.1],      # 第七笔起始点(F13,G13)
        control_point=[-0.13, 0.45, 0.3],  # 第六笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 14. 书写第七笔
    TrajectorySegment(
        start_point=[-0.13, 0.45, 0.1],    # F13, G13
        end_point=[-0.13, 0.41, 0.1],      # F14, G14
        interp_type='linear',
        duration=1.0
    ),
    
    # 15. 移动到第八笔起始点
    TrajectorySegment(
        start_point=[-0.13, 0.41, 0.1],    # 第七笔终止点
        end_point=[-0.13, 0.41, 0.1],      # 第八笔起始点(F15,G15)
        control_point=[-0.13, 0.41, 0.3],  # 第七笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 16. 书写第八笔
    TrajectorySegment(
        start_point=[-0.13, 0.41, 0.1],    # F15, G15
        end_point=[-0.15, 0.41, 0.1],      # F16, G16
        interp_type='linear',
        duration=1.0
    ),
    
    # 17. 移动到第九笔起始点
    TrajectorySegment(
        start_point=[-0.15, 0.41, 0.1],    # 第八笔终止点
        end_point=[-0.19, 0.44, 0.1],      # 第九笔起始点(F17,G17)
        control_point=[-0.15, 0.41, 0.3],  # 第八笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 18. 书写第九笔（最后一笔）
    TrajectorySegment(
        start_point=[-0.19, 0.44, 0.1],    # F17, G17
        end_point=[-0.08, 0.44, 0.1],      # F18, G18
        interp_type='linear',
        duration=1.0
    ),
    
    # ========== 移动到文字起始点（从"李"字最后一笔到"文"字第一笔） ==========
    TrajectorySegment(
        start_point=[-0.08, 0.44, 0.1],    # "李"字最后一笔终止点
        end_point=[-0.01, 0.52, 0.1],      # "文"字第一笔起始点(F19,G19)
        control_point=[-0.08, 0.44, 0.3],  # "李"字最后一笔终止点正上方z=0.3
        interp_type='bezier',
        duration=2.0
    ),
    
    # ========== 文 字 ==========
    # 19. 书写"文"字第一笔
    TrajectorySegment(
        start_point=[-0.01, 0.52, 0.1],    # F19, G19
        end_point=[0.01, 0.5, 0.1],        # F20, G20
        interp_type='linear',
        duration=1.0
    ),
    
    # 20. 移动到"文"字第二笔起始点
    TrajectorySegment(
        start_point=[0.01, 0.5, 0.1],      # "文"字第一笔终止点
        end_point=[-0.055, 0.49, 0.1],     # "文"字第二笔起始点(F21,G21)
        control_point=[0.01, 0.5, 0.3],    # "文"字第一笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 21. 书写"文"字第二笔
    TrajectorySegment(
        start_point=[-0.055, 0.49, 0.1],   # F21, G21
        end_point=[0.055, 0.49, 0.1],      # F22, G22
        interp_type='linear',
        duration=1.0
    ),
    
    # 22. 移动到"文"字第三笔起始点
    TrajectorySegment(
        start_point=[0.055, 0.49, 0.1],    # "文"字第二笔终止点
        end_point=[0.025, 0.49, 0.1],      # "文"字第三笔起始点(F23,G23)
        control_point=[0.055, 0.49, 0.3],  # "文"字第二笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 23. 书写"文"字第三笔
    TrajectorySegment(
        start_point=[0.025, 0.49, 0.1],    # F23, G23
        end_point=[-0.05, 0.4, 0.1],       # F24, G24
        interp_type='linear',
        duration=1.0
    ),
    
    # 24. 移动到"文"字第四笔起始点
    TrajectorySegment(
        start_point=[-0.05, 0.4, 0.1],     # "文"字第三笔终止点
        end_point=[-0.025, 0.49, 0.1],     # "文"字第四笔起始点(F25,G25)
        control_point=[-0.05, 0.4, 0.3],   # "文"字第三笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 25. 书写"文"字第四笔
    TrajectorySegment(
        start_point=[-0.025, 0.49, 0.1],   # F25, G25
        end_point=[0.05, 0.4, 0.1],        # F26, G26
        interp_type='linear',
        duration=1.0
    ),
    
    # ========== 移动到恒字起始点（从"文"字最后一笔到"恒"字第一笔） ==========
    TrajectorySegment(
        start_point=[0.05, 0.4, 0.1],      # "文"字最后一笔终止点
        end_point=[0.095, 0.52, 0.1],      # "恒"字第一笔起始点(F27,G27)
        control_point=[0.05, 0.4, 0.3],    # "文"字最后一笔终止点正上方z=0.3
        interp_type='bezier',
        duration=2.0
    ),
    
    # ========== 恒 字 ==========
    # 26. 书写"恒"字第一笔
    TrajectorySegment(
        start_point=[0.095, 0.52, 0.1],    # F27, G27
        end_point=[0.095, 0.4, 0.1],       # F28, G28
        interp_type='linear',
        duration=1.0
    ),
    
    # 27. 移动到"恒"字第二笔起始点
    TrajectorySegment(
        start_point=[0.095, 0.4, 0.1],     # "恒"字第一笔终止点
        end_point=[0.08, 0.485, 0.1],      # "恒"字第二笔起始点(F29,G29)
        control_point=[0.095, 0.4, 0.3],   # "恒"字第一笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 28. 书写"恒"字第二笔
    TrajectorySegment(
        start_point=[0.08, 0.485, 0.1],    # F29, G29
        end_point=[0.075, 0.465, 0.1],     # F30, G30
        interp_type='linear',
        duration=1.0
    ),
    
    # 29. 移动到"恒"字第三笔起始点
    TrajectorySegment(
        start_point=[0.075, 0.465, 0.1],   # "恒"字第二笔终止点
        end_point=[0.105, 0.485, 0.1],     # "恒"字第三笔起始点(F31,G31)
        control_point=[0.075, 0.465, 0.3], # "恒"字第二笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 30. 书写"恒"字第三笔
    TrajectorySegment(
        start_point=[0.105, 0.485, 0.1],   # F31, G31
        end_point=[0.112, 0.48, 0.1],      # F32, G32
        interp_type='linear',
        duration=1.0
    ),
    
    # 31. 移动到"恒"字第四笔起始点
    TrajectorySegment(
        start_point=[0.112, 0.48, 0.1],    # "恒"字第三笔终止点
        end_point=[0.115, 0.51, 0.1],      # "恒"字第四笔起始点(F33,G33)
        control_point=[0.112, 0.48, 0.3],  # "恒"字第三笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 32. 书写"恒"字第四笔
    TrajectorySegment(
        start_point=[0.115, 0.51, 0.1],    # F33, G33
        end_point=[0.185, 0.51, 0.1],      # F34, G34
        interp_type='linear',
        duration=1.0
    ),
    
    # 33. 移动到"恒"字第五笔起始点
    TrajectorySegment(
        start_point=[0.185, 0.51, 0.1],    # "恒"字第四笔终止点
        end_point=[0.125, 0.475, 0.1],     # "恒"字第五笔起始点(F35,G35)
        control_point=[0.185, 0.51, 0.3],  # "恒"字第四笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 34. 书写"恒"字第五笔
    TrajectorySegment(
        start_point=[0.125, 0.475, 0.1],   # F35, G35
        end_point=[0.125, 0.44, 0.1],      # F36, G36
        interp_type='linear',
        duration=1.0
    ),
    
    # 35. 移动到"恒"字第六笔起始点
    TrajectorySegment(
        start_point=[0.125, 0.44, 0.1],    # "恒"字第五笔终止点
        end_point=[0.175, 0.475, 0.1],     # "恒"字第六笔起始点(F37,G37)
        control_point=[0.125, 0.44, 0.3],  # "恒"字第五笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 36. 书写"恒"字第六笔
    TrajectorySegment(
        start_point=[0.175, 0.475, 0.1],   # F37, G37
        end_point=[0.175, 0.44, 0.1],      # F38, G38
        interp_type='linear',
        duration=1.0
    ),
    
    # 37. 移动到"恒"字第七笔起始点
    TrajectorySegment(
        start_point=[0.175, 0.44, 0.1],    # "恒"字第六笔终止点
        end_point=[0.125, 0.475, 0.1],     # "恒"字第七笔起始点(F39,G39)
        control_point=[0.175, 0.44, 0.3],  # "恒"字第六笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 38. 书写"恒"字第七笔（横）
    TrajectorySegment(
        start_point=[0.125, 0.475, 0.1],   # F39, G39
        end_point=[0.175, 0.475, 0.1],     # F40, G40
        interp_type='linear',
        duration=1.0
    ),
    
    # 39. 移动到"恒"字第八笔起始点
    TrajectorySegment(
        start_point=[0.175, 0.475, 0.1],   # "恒"字第七笔终止点
        end_point=[0.125, 0.4575, 0.1],    # "恒"字第八笔起始点(F41,G41)
        control_point=[0.175, 0.475, 0.3], # "恒"字第七笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 40. 书写"恒"字第八笔（横）
    TrajectorySegment(
        start_point=[0.125, 0.4575, 0.1],  # F41, G41
        end_point=[0.175, 0.4575, 0.1],    # F42, G42
        interp_type='linear',
        duration=1.0
    ),
    
    # 41. 移动到"恒"字第九笔起始点
    TrajectorySegment(
        start_point=[0.175, 0.4575, 0.1],  # "恒"字第八笔终止点
        end_point=[0.125, 0.44, 0.1],      # "恒"字第九笔起始点(F43,G43)
        control_point=[0.175, 0.4575, 0.3],# "恒"字第八笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 42. 书写"恒"字第九笔（横）
    TrajectorySegment(
        start_point=[0.125, 0.44, 0.1],    # F43, G43
        end_point=[0.175, 0.44, 0.1],      # F44, G44
        interp_type='linear',
        duration=1.0
    ),
    
    # 43. 移动到"恒"字第十笔起始点
    TrajectorySegment(
        start_point=[0.175, 0.44, 0.1],    # "恒"字第九笔终止点
        end_point=[0.115, 0.415, 0.1],     # "恒"字第十笔起始点(F45,G45)
        control_point=[0.175, 0.44, 0.3],  # "恒"字第九笔终止点正上方z=0.3
        interp_type='bezier',
        duration=1.5
    ),
    
    # 44. 书写"恒"字第十笔（最后一笔）
    TrajectorySegment(
        start_point=[0.115, 0.415, 0.1],   # F45, G45
        end_point=[0.185, 0.415, 0.1],     # F46, G46
        interp_type='linear',
        duration=1.0
    )
]

# 轨迹执行状态
current_segment_index = 0
segment_start_time = 0.0
X_ref = trajectory_segments[0].start_point  # 初始参考位置与首个轨迹段起点一致

######################################
### INTERPOLATION FUNCTIONS ###
def LinearInterpolate(q0, q1, t, t_total):
    u = t / t_total
    u = np.clip(u, 0, 1)
    # 正弦缓动，提高平滑性
    u = np.sin(u * np.pi / 2)
    return q0 + u * (q1 - q0)

def QuadBezierInterpolate(q0, q1, q2, t, t_total):
    u = t / t_total
    u = np.clip(u, 0, 1)
    # 正弦缓动优化
    u = np.sin(u * np.pi / 2)
    return (1 - u)**2 * q0 + 2 * u * (1 - u) * q1 + u**2 * q2

def get_reference_position(current_time):
    global current_segment_index, segment_start_time, X_ref, trajectory_segments
    
    if current_segment_index >= len(trajectory_segments):
        return trajectory_segments[-1].end_point
    
    current_segment = trajectory_segments[current_segment_index]
    segment_elapsed_time = current_time - segment_start_time
    
    # 轨迹段切换逻辑优化
    if segment_elapsed_time >= current_segment.duration:
        current_segment.completed = True
        X_ref = current_segment.end_point
        current_segment_index += 1
        
        if current_segment_index < len(trajectory_segments):
            segment_start_time = current_time
            current_segment = trajectory_segments[current_segment_index]
            X_ref = current_segment.start_point
    
    # 计算当前参考位置
    if current_segment.interp_type == 'linear':
        X_ref = LinearInterpolate(
            current_segment.start_point,
            current_segment.end_point,
            segment_elapsed_time,
            current_segment.duration
        )
    elif current_segment.interp_type == 'bezier' and current_segment.control_point is not None:
        X_ref = QuadBezierInterpolate(
            current_segment.start_point,
            current_segment.control_point,
            current_segment.end_point,
            segment_elapsed_time,
            current_segment.duration
        )
    
    return X_ref
############################################

while not glfw.window_should_close(window):
    time_prev = data.time

    # 固定帧率更新，移除手动时间叠加
    while (data.time - time_prev < 1.0/60.0):
        # Store trajectory（核心：z≤0.11才存储轨迹点，避免关键字冲突）
        mj_end_eff_pos = data.site_xpos[0]
        # 仅当z≤写字阈值时，存储轨迹点（形成字迹）
        if mj_end_eff_pos[2] <= WRITE_Z_THRESHOLD:
            if len(traj_points) == 0:
                traj_points.append(mj_end_eff_pos.copy())
            else:
                # 计算与上一个点的距离，去重
                last_point = traj_points[-1]
                dist = np.linalg.norm(mj_end_eff_pos - last_point)
                if dist >= MIN_DISTANCE:
                    traj_points.append(mj_end_eff_pos.copy())
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)
            
        cur_q_pos = data.qpos.copy()
        X_ref = get_reference_position(data.time)
        cur_ctrl = IK_controller(model, data, X_ref, cur_q_pos)
        data.ctrl[:] = cur_ctrl
        
        # 仅通过mj_step更新时间
        mj.mj_step(model, data)

    if (data.time >= simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if (print_camera_config==1):
        print('cam.azimuth = ', cam.azimuth, '\n', 'cam.elevation = ', cam.elevation, '\n', 'cam.distance = ', cam.distance)
        print('cam.lookat = np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # 修正：改用球体绘制轨迹（避开from关键字，无语法错误）
    for j in range(len(traj_points)):
        if scene.ngeom >= scene.maxgeom:
            break

        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        
        # 用球体绘制轨迹点，无关键字冲突
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.002, 0.002, 0.002])
        geom.pos[:] = traj_points[j]
        geom.mat[:] = np.eye(3)
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0
        
    # 轨迹段标记可视化
    for i, segment in enumerate(trajectory_segments):
        if scene.ngeom >= scene.maxgeom:
            break
            
        # 起点（绿色）
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = np.array([0.0, 1.0, 0.0, 1.0])
        geom.size[:] = np.array([0.005, 0.005, 0.005])
        geom.pos[:] = segment.start_point
        geom.mat[:] = np.eye(3)
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0
        
        if scene.ngeom >= scene.maxgeom:
            break
            
        # 终点（蓝色）
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = np.array([0.0, 0.0, 1.0, 1.0])
        geom.size[:] = np.array([0.005, 0.005, 0.005])
        geom.pos[:] = segment.end_point
        geom.mat[:] = np.eye(3)
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0
        
        if segment.interp_type == 'bezier' and segment.control_point is not None:
            if scene.ngeom >= scene.maxgeom:
                break
                
            # 控制点（黄色）
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            geom.type = mj.mjtGeom.mjGEOM_SPHERE
            geom.rgba[:] = np.array([1.0, 1.0, 0.0, 1.0])
            geom.size[:] = np.array([0.004, 0.004, 0.004])
            geom.pos[:] = segment.control_point
            geom.mat[:] = np.eye(3)
            geom.dataid = -1
            geom.segid = -1
            geom.objtype = 0
            geom.objid = 0
    
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()