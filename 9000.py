import os
import csv
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

# ============================================================
# 0) 关键参数集中区（你以后主要改这里）
# ============================================================
XML_PATH = "../../models/universal_robots_ur5e/scene.xml"
CSV_PATH = "./trajectory_lagrange_rect.csv"

SIM_TIME = 1000.0
FPS = 60.0

# 记录字迹的条件：z <= write_limit 才记录
WRITE_LIMIT = 0.098
MIN_DIST = 0.001
NO_RECORD_TIME = 0.6

# 末端 site（请确认 0 是笔尖 site）
EE_SITE_ID = 0

# IK 初始关节（你给的"写字友好"版本）
INIT_QPOS = np.array([-1.57, -1.20, 2.05, -2.45, -1.57, 0.0], dtype=float)

# 相机
CAM_AZIMUTH = 179.83
CAM_ELEVATION = 87.1633
CAM_DISTANCE = 2.22
CAM_LOOKAT = np.array([0.29723477517870245, 0.28277006411151073, 0.6082647377843177], dtype=float)

# 轨迹可视化
MAX_TRAJ = 5000
TRJ_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
TRJ_SPHERE_SIZE = 0.002

# 关节限位（UR5e 6关节）
UR5E_JOINT_LIMITS = np.array([[-np.pi, np.pi]] * 6, dtype=float)

# ====== 你这版"好的控制器"参数（6D DLS + 笔尖朝下） ======
DAMP = 1e-2
POS_GAIN = 1.2
ROT_GAIN = 0.6
DQ_LIMIT = 0.12


# ============================================================
# 1) 轨迹段：CSV 读取（最多四点：start + c1? + c2? + end）
# ============================================================
class TrajectorySegment:
    def __init__(self, start_point, end_point, interp_type, duration, control_points=None):
        self.start_point = np.array(start_point, dtype=float)
        self.end_point = np.array(end_point, dtype=float)
        self.interp_type = str(interp_type).strip()
        self.duration = float(duration)
        # catmull_rom 用：曲线上的中间点（不含 start/end），最多2个
        self.control_points = [] if control_points is None else [np.array(p, dtype=float) for p in control_points]


def _parse_float_or_none(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None

def _must_float(row, key, row_i):
    # 处理可能的空格问题
    v = row.get(key)
    if v is None:
        # 尝试去掉末尾的空格和逗号
        for k in row.keys():
            if k and k.strip() == key:
                v = row[k]
                break
    
    if v is None:
        # 打印可用的键以调试
        print(f"Row {row_i}: Available keys: {list(row.keys())}")
        raise ValueError(f"Row {row_i}: missing required field '{key}'")
    
    v = _parse_float_or_none(v)
    if v is None:
        raise ValueError(f"Row {row_i}: field '{key}' cannot be parsed as float: {row.get(key)}")
    return v

def _parse_point3_optional(row, xk, yk, zk, row_i):
    # 处理可能的空格问题
    x = None
    y = None
    z = None
    
    # 尝试不同的键名格式
    for k in row.keys():
        if k and k.strip() == xk:
            x = _parse_float_or_none(row[k])
        elif k and k.strip() == yk:
            y = _parse_float_or_none(row[k])
        elif k and k.strip() == zk:
            z = _parse_float_or_none(row[k])
    
    if x is None and y is None and z is None:
        return None
    if x is None or y is None or z is None:
        # 打印调试信息
        print(f"Row {row_i}: Incomplete point ({xk},{yk},{zk}) - x={x}, y={y}, z={z}")
        print(f"Row {row_i}: Available keys: {list(row.keys())}")
        raise ValueError(f"Row {row_i}: incomplete point ({xk},{yk},{zk})")
    return [x, y, z]

def load_segments_from_csv(path: str):
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames:
            reader.fieldnames = [field.strip() for field in reader.fieldnames if field]
            print(f"CSV headers found: {reader.fieldnames}")

        required = ["interp_type","duration","sx","sy","sz","ex","ey","ez"]
        for rf in required:
            if rf not in reader.fieldnames:
                raise ValueError(f"CSV missing required field: {rf}")

        for i, row in enumerate(reader):
            cleaned_row = {k.strip(): v for k, v in row.items() if k}

            interp = str(cleaned_row.get("interp_type", "linear")).strip()
            duration = _must_float(cleaned_row, "duration", i)

            sx = _must_float(cleaned_row, "sx", i)
            sy = _must_float(cleaned_row, "sy", i)
            sz = _must_float(cleaned_row, "sz", i)

            ex = _must_float(cleaned_row, "ex", i)
            ey = _must_float(cleaned_row, "ey", i)
            ez = _must_float(cleaned_row, "ez", i)

            # 解析任意数量的 cN 控制点（严格在起终点之后）
            cps = []
            n = 1
            while True:
                keyx, keyy, keyz = f"c{n}x", f"c{n}y", f"c{n}z"
                if keyx in cleaned_row or keyy in cleaned_row or keyz in cleaned_row:
                    pt = _parse_point3_optional(cleaned_row, keyx, keyy, keyz, i)
                    if pt is not None:
                        cps.append(pt)
                    n += 1
                else:
                    break

            if interp == "catmull_rom":
                interp = "lagrange"  # 统一替换为新的插值法

            segs.append(TrajectorySegment(
                start_point=[sx, sy, sz],
                end_point=[ex, ey, ez],
                interp_type=interp,
                duration=duration,
                control_points=cps
            ))

            print(f"Loaded segment {i}: {interp} start→end with {len(cps)} control points")

    if not segs:
        raise ValueError("trajectory CSV is empty")

    print(f"Successfully loaded {len(segs)} trajectory segments")
    return segs





# ============================================================
# 1.5) 插值：linear + Lagrange（支持任意数量控制点）
# ============================================================

def ease_sine(u: float) -> float:
    """
    平滑归一化函数，将 [0,1] 映射为更自然的插值曲线。
    """
    u = np.clip(u, 0.0, 1.0)
    return np.sin(u * np.pi / 2.0)


def linear_interp(p0: np.ndarray, p1: np.ndarray, t: float, T: float) -> np.ndarray:
    """
    线性插值（带 ease_sine 平滑）。
    p0, p1: 起点和终点 (3,)
    t: 当前时间
    T: 总时长
    """
    u = ease_sine(t / max(T, 1e-9))
    return p0 + u * (p1 - p0)


def lagrange_interp(points: np.ndarray, u: float) -> np.ndarray:
    """
    拉格朗日插值法，支持任意数量的控制点。
    points: (m,3) 点集，m>=2
    u: [0,1] 的归一化参数
    """
    points = np.asarray(points, dtype=float)
    m = points.shape[0]

    # 节点均匀分布在 [0,1]
    x = np.linspace(0, 1, m)
    target = u

    result = np.zeros(3, dtype=float)
    for j in range(m):
        Lj = 1.0
        for k in range(m):
            if k != j:
                Lj *= (target - x[k]) / (x[j] - x[k])
        result += Lj * points[j]
    return result


def lagrange_interp_segment(start: np.ndarray,
                            control_points: list,
                            end: np.ndarray,
                            t: float,
                            T: float) -> np.ndarray:
    """
    通用插值接口：传入起点 + 任意数量控制点 + 终点。
    """
    u = ease_sine(t / max(T, 1e-9))
    pts = [start] + list(control_points) + [end]
    return lagrange_interp(np.vstack(pts), u)




class TrajectoryPlayer:
    def __init__(self, segments):
        self.segments = segments
        self.idx = 0
        self.t0 = 0.0

    def reset(self, t):
        self.idx = 0
        self.t0 = t

    def sample(self, t):
        if self.idx >= len(self.segments):
            return self.segments[-1].end_point.copy()

        seg = self.segments[self.idx]
        dt = t - self.t0

        # 如果当前段完成，切换到下一段
        while dt >= seg.duration and self.idx < len(self.segments) - 1:
            dt -= seg.duration
            self.idx += 1
            self.t0 = t - dt
            seg = self.segments[self.idx]

        # 根据插值类型选择方法
        if seg.interp_type == "linear":
            return linear_interp(seg.start_point, seg.end_point, dt, seg.duration)

        if seg.interp_type == "lagrange":
            return lagrange_interp_segment(
                seg.start_point,
                seg.control_points,  # 控制点都在后面
                seg.end_point,
                dt,
                seg.duration
            )


        # 默认回退到线性插值
        return linear_interp(seg.start_point, seg.end_point, dt, seg.duration)



# ============================================================
# 2) 控制器：按你"好的版本"做 6D DLS IK（位置+姿态，笔尖朝下）
# ============================================================
def IK_controller(model, data, X_ref, q_pos):
    position_Q = data.site_xpos[EE_SITE_ID].copy()
    site_xmat = data.site_xmat[EE_SITE_ID].reshape(3, 3).copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, EE_SITE_ID)

    # 位置误差
    dX_pos = (X_ref - position_Q) * POS_GAIN

    # 姿态误差：让末端z轴对齐世界 -Z（笔尖朝下）
    z_world_des = np.array([0.0, 0.0, -1.0], dtype=float)
    z_site = site_xmat[:, 2]
    dX_rot = np.cross(z_site, z_world_des) * ROT_GAIN

    # 6D
    J = np.vstack((jacp, jacr))
    dX = np.hstack((dX_pos, dX_rot))

    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (DAMP**2) * np.eye(6), dX)

    dq = np.clip(dq, -DQ_LIMIT, DQ_LIMIT)
    q_new = q_pos + dq

    # 关节限位
    for i in range(min(6, len(q_new))):
        q_new[i] = np.clip(q_new[i], UR5E_JOINT_LIMITS[i, 0], UR5E_JOINT_LIMITS[i, 1])

    return q_new


# ============================================================
# 3) 轨迹记录 + 绘制辅助
# ============================================================
def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    g.type = mj.mjtGeom.mjGEOM_SPHERE
    g.rgba[:] = rgba
    g.size[:] = np.array([radius, radius, radius], dtype=float)
    g.pos[:] = pos
    g.mat[:] = np.eye(3)
    g.dataid = -1
    g.segid = -1
    g.objtype = 0
    g.objid = 0


# ============================================================
# 4) GLFW 交互
# ============================================================
class UI:
    def __init__(self, model, scene, cam):
        self.model = model
        self.scene = scene
        self.cam = cam
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

    def keyboard(self, window, key, scancode, act, mods):
        pass

    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or \
                    (glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05*yoffset, self.scene, self.cam)


# ============================================================
# 5) Main
# ============================================================
def main():
    here = os.path.dirname(__file__)
    xml_abspath = os.path.join(here, XML_PATH)
    csv_abspath = os.path.join(here, CSV_PATH)

    if not os.path.exists(csv_abspath):
        raise FileNotFoundError(
            f"CSV not found: {csv_abspath}\n"
            f"请先运行 export_trajectory_to_csv.py 生成它，或把 CSV 放到该路径。"
        )

    model = mj.MjModel.from_xml_path(xml_abspath)
    data = mj.MjData(model)

    data.qpos[:] = INIT_QPOS
    mj.mj_forward(model, data)

    segments = load_segments_from_csv(csv_abspath)
    player = TrajectoryPlayer(segments)
    player.reset(data.time)

    glfw.init()
    window = glfw.create_window(1920, 1080, "Writer", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    cam.distance = CAM_DISTANCE
    cam.lookat = CAM_LOOKAT.copy()

    ui = UI(model, scene, cam)
    glfw.set_cursor_pos_callback(window, ui.mouse_move)
    glfw.set_mouse_button_callback(window, ui.mouse_button)
    glfw.set_scroll_callback(window, ui.scroll)

    traj_points = []
    dt_frame = 1.0 / FPS

    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev) < dt_frame:
            ee = data.site_xpos[EE_SITE_ID].copy()
            if data.time >= NO_RECORD_TIME and ee[2] <= WRITE_LIMIT:
                if not traj_points:
                    traj_points.append(ee)
                else:
                    if np.linalg.norm(ee - traj_points[-1]) >= MIN_DIST:
                        traj_points.append(ee)
                        if len(traj_points) > MAX_TRAJ:
                            traj_points.pop(0)

            X_ref = player.sample(data.time)

            q = data.qpos.copy()
            q_des = IK_controller(model, data, X_ref, q)

            data.ctrl[:] = q_des
            mj.mj_step(model, data)

            if data.time >= SIM_TIME:
                glfw.terminate()
                return

        w, h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, w, h)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

        for p in traj_points:
            add_sphere(scene, p, TRJ_SPHERE_SIZE, TRJ_RGBA)

        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()