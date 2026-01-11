from scipy.interpolate import splprep, splev
import os
import csv
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt

# ============================================================
# 0) 关键参数集中区
# ============================================================
XML_PATH = "../../models/universal_robots_ur5e/scene.xml"
LETTERS_DIR = "./letters"  # 字母轨迹文件目录

sample_per_s = 500  # 每秒采样点数

SIM_TIME = 180.0  # 仿真总时长（秒）
FPS = 8 # 显示帧率

# 记录字迹的条件
WRITE_LIMIT = 0.1
MIN_DIST = 1e-3
NO_RECORD_TIME = 1e-3

# 显示轨迹的z坐标阈值 - 只显示z小于此值的轨迹
DISPLAY_Z_THRESHOLD = 0.098 # 新增：只显示z坐标小于此值的轨迹点

# 末端 site
EE_SITE_ID = 0

# IK 初始关节
INIT_QPOS = np.array([-1.57, -1.20, 2.05, -2.45, -1.57, 0.0], dtype=float)

# 相机
CAM_AZIMUTH = 179.83
CAM_ELEVATION = 87.1633
CAM_DISTANCE = 2.22
CAM_LOOKAT = np.array([0.29723477517870245, 0.28277006411151073, 0.6082647377843177], dtype=float)

# 轨迹可视化
MAX_TRAJ = 500000
TRJ_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
TRJ_SPHERE_SIZE = 0.002

# 关节限位（UR5e 6关节）
UR5E_JOINT_LIMITS = np.array([[-np.pi, np.pi]] * 6, dtype=float)

# IK 参数
DAMP = 1e-3
POS_GAIN = 2
ROT_GAIN = 1
DQ_LIMIT = 0.2

# 最终关节角度
FINAL_QPOS = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0], dtype=float)

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

def _parse_point3_optional(row, xk, yk, zk, row_i, allow_partial=False):
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
    
    # 如果允许部分为空，则检查是否有至少一个坐标不为空
    if allow_partial:
        if x is None and y is None and z is None:
            return None
        # 如果某些坐标是None，使用默认值0
        if x is None:
            x = 0.0
        if y is None:
            y = 0.0
        if z is None:
            z = 0.1  # 默认z坐标为0.1
        return [x, y, z]
    else:
        if x is None and y is None and z is None:
            return None
        if x is None or y is None or z is None:
            # 打印调试信息
            print(f"Row {row_i}: Incomplete point ({xk},{yk},{zk}) - x={x}, y={y}, z={z}")
            print(f"Row {row_i}: Available keys: {list(row.keys())}")
            raise ValueError(f"Row {row_i}: incomplete point ({xk},{yk},{zk})")
        return [x, y, z]

def load_segments_from_csv(path: str, x_offset: float = 0.0, y_offset: float = 0.0):
    """加载轨迹段，可选的x轴和y轴偏移"""
    segs = []
    
    # 尝试不同的编码方式读取CSV文件
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            with open(path, "r", encoding=encoding) as f:
                # 读取第一行来检查是否有BOM
                first_line = f.readline()
                # 重置文件指针
                f.seek(0)
                
                reader = csv.DictReader(f)
                
                # 检查是否有字段
                if not reader.fieldnames:
                    continue
                    
                # 清理字段名：去除BOM和空格
                cleaned_fieldnames = []
                for field in reader.fieldnames:
                    # 去除BOM字符和空白字符
                    cleaned = field.strip().replace('\ufeff', '')
                    cleaned_fieldnames.append(cleaned)
                
                reader.fieldnames = cleaned_fieldnames
                
                print(f"成功以 {encoding} 编码读取文件，字段名: {reader.fieldnames}")
                
                # 尝试不同的字段名
                type_fields = ["type", "interp_type", "Type", "Interp_type"]
                duration_fields = ["duration", "Duration"]
                sx_fields = ["sx", "Sx"]
                sy_fields = ["sy", "Sy"]
                sz_fields = ["sz", "Sz"]
                ex_fields = ["ex", "Ex"]
                ey_fields = ["ey", "Ey"]
                ez_fields = ["ez", "Ez"]
                
                # 查找实际的字段名
                def find_field(field_list, default=None):
                    for f in field_list:
                        if f in reader.fieldnames:
                            return f
                    return default
                
                type_field = find_field(type_fields, "type")
                duration_field = find_field(duration_fields, "duration")
                sx_field = find_field(sx_fields, "sx")
                sy_field = find_field(sy_fields, "sy")
                sz_field = find_field(sz_fields, "sz")
                ex_field = find_field(ex_fields, "ex")
                ey_field = find_field(ey_fields, "ey")
                ez_field = find_field(ez_fields, "ez")
                
                required = [type_field, duration_field, sx_field, sy_field, sz_field, ex_field, ey_field, ez_field]
                
                # 检查必需字段
                missing_fields = []
                for rf in required:
                    if rf is None:
                        missing_fields.append(rf)
                
                if missing_fields:
                    print(f"编码 {encoding} 下缺少字段: {missing_fields}")
                    continue
                
                for i, row in enumerate(reader):
                    cleaned_row = {k.strip(): v for k, v in row.items() if k}

                    # 使用找到的字段名
                    interp = str(cleaned_row.get(type_field, "linear")).strip()
                    duration = _must_float(cleaned_row, duration_field, i)

                    # 加载并应用x轴和y轴偏移
                    sx = _must_float(cleaned_row, sx_field, i) + x_offset
                    sy = _must_float(cleaned_row, sy_field, i) + y_offset
                    sz = _must_float(cleaned_row, sz_field, i)

                    ex = _must_float(cleaned_row, ex_field, i) + x_offset
                    ey = _must_float(cleaned_row, ey_field, i) + y_offset
                    ez = _must_float(cleaned_row, ez_field, i)

                    # 解析任意数量的 cN 控制点（严格在起终点之后）
                    cps = []
                    n = 1
                    while True:
                        # 尝试不同大小写的字段名
                        keyx_variants = [f"c{n}x", f"c{n}X", f"C{n}x", f"C{n}X"]
                        keyy_variants = [f"c{n}y", f"c{n}Y", f"C{n}y", f"C{n}Y"]
                        keyz_variants = [f"c{n}z", f"c{n}Z", f"C{n}z", f"C{n}Z"]
                        
                        found = False
                        for keyx in keyx_variants:
                            for keyy in keyy_variants:
                                for keyz in keyz_variants:
                                    if any(k in cleaned_row for k in [keyx, keyy, keyz]):
                                        pt = _parse_point3_optional(cleaned_row, keyx, keyy, keyz, i, allow_partial=True)
                                        if pt is not None:
                                            # 应用x轴和y轴偏移到控制点
                                            pt[0] += x_offset
                                            pt[1] += y_offset
                                            cps.append(pt)
                                            found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        
                        if not found:
                            break
                        n += 1

                    segs.append(TrajectorySegment(
                        start_point=[sx, sy, sz],
                        end_point=[ex, ey, ez],
                        interp_type=interp,
                        duration=duration,
                        control_points=cps
                    ))
                    print(f"  段 {i}: {interp}, 时长 {duration}, 起点 ({sx:.3f}, {sy:.3f}, {sz:.3f}), 终点 ({ex:.3f}, {ey:.3f}, {ez:.3f}), 控制点 {len(cps)}个")

            # 如果成功读取了数据，跳出循环
            if segs:
                print(f"成功从 {path} 加载了 {len(segs)} 个轨迹段")
                return segs
                
        except Exception as e:
            print(f"以 {encoding} 编码读取文件失败: {e}")
            continue
    
    # 如果所有编码都失败了，抛出异常
    raise ValueError(f"无法以任何支持的编码读取文件: {path}")

# ============================================================
# 1.5) 插值：linear + Lagrange（支持任意数量控制点，无平滑）
# ============================================================
def linear_interp(p0: np.ndarray, p1: np.ndarray, t: float, T: float) -> np.ndarray:
    """
    线性插值（无平滑）。
    p0, p1: 起点和终点 (3,)
    t: 当前时间
    T: 总时长
    """
    u = np.clip(t / max(T, 1e-9), 0.0, 1.0)
    return p0 + u * (p1 - p0)

def bspline_interp(points: np.ndarray, u: float, degree: int = 3) -> np.ndarray:
    """
    B样条插值，支持任意数量控制点。
    points: (m,3) 点集，m>=2
    u: [0,1] 的归一化参数
    degree: 样条阶数，默认3（三次B样条）
    """
    points = np.asarray(points, dtype=float).T  # shape (3,m)
    # 构造样条
    tck, _ = splprep(points, s=0, k=min(degree, len(points.T)-1))
    # 计算插值点
    result = np.array(splev(u, tck))
    return result

def quadratic_bezier_interp(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, u: float) -> np.ndarray:
    return (1 - u)**2 * p0 + 2 * (1 - u) * u * p1 + u**2 * p2

# ============================================================
class TrajectoryPlayer:
    def __init__(self, segments, n_samples=sample_per_s):
        self.segments = segments
        self.idx = 0
        self.t0 = 0.0
        self.arc_tables = []
        self.last_point = None

        for seg in segments:
            pts = [seg.start_point] + seg.control_points + [seg.end_point]
            pts = np.vstack(pts)

            us = np.linspace(0, 1, n_samples)
            if seg.interp_type.lower() == "lagrange":
                curve = np.array([bspline_interp(pts, u) for u in us])
            elif seg.interp_type.lower() == "bspline":
                if len(seg.control_points) < 1:
                    print(f"Warning: segment {self.idx} 标记为 bspline 但没有控制点，退化为线性插值")
                    curve = np.array([linear_interp(seg.start_point, seg.end_point, u, 1.0) for u in us])
                else:
                    p0 = seg.start_point
                    p1 = seg.control_points[0]
                    p2 = seg.end_point
                    curve = np.array([quadratic_bezier_interp(p0, p1, p2, u) for u in us])
            else:
                curve = np.array([linear_interp(seg.start_point, seg.end_point, u, 1.0) for u in us])

            dists = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            s = np.concatenate(([0], np.cumsum(dists)))
            s /= s[-1] if s[-1] > 1e-9 else 1.0

            self.arc_tables.append((us, curve, s))

    def reset(self, t):
        self.idx = 0
        self.t0 = t
        self.last_point = None

    def sample(self, t):
        if self.idx >= len(self.segments):
            return self.segments[-1].end_point.copy()

        seg = self.segments[self.idx]
        dt = t - self.t0

        while dt >= seg.duration and self.idx < len(self.segments) - 1:
            dt -= seg.duration
            self.idx += 1
            self.t0 = t - dt
            seg = self.segments[self.idx]

        u_time = np.clip(dt / seg.duration, 0.0, 1.0)
        us, curve, s = self.arc_tables[self.idx]
        idx = np.searchsorted(s, u_time)
        idx = np.clip(idx, 0, len(us)-1)

        point = curve[idx]
        if self.last_point is None:
            self.last_point = point
        else:
            self.last_point = 0.9 * self.last_point + 0.1 * point
        return self.last_point

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
    letters_dir = os.path.join(here, LETTERS_DIR)
    
    # 在Windows系统上，尝试使用短路径名避免中文字符问题
    if os.name == 'nt':
        try:
            # 尝试转换为短路径
            import ctypes
            from ctypes import wintypes
            
            # 定义函数
            GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            GetShortPathNameW.restype = wintypes.DWORD
            
            def get_short_path_name(long_name):
                # 分配缓冲区
                output_buf_size = 4096
                output_buf = ctypes.create_unicode_buffer(output_buf_size)
                # 调用函数
                GetShortPathNameW(long_name, output_buf, output_buf_size)
                return output_buf.value
            
            xml_abspath = get_short_path_name(xml_abspath)
            print(f"使用短路径访问XML文件: {xml_abspath}")
        except Exception as e:
            print(f"无法获取短路径，使用原路径: {e}")
    
    print(f"XML路径: {xml_abspath}")
    print(f"字母目录路径: {letters_dir}")
    
    # 确保字母轨迹目录存在
    if not os.path.exists(letters_dir):
        os.makedirs(letters_dir)
        print(f"创建了字母目录: {letters_dir}")
        print("请将字母轨迹文件 (如 A.csv, B.csv, J.csv 等) 放入此目录")
        return

    # 获取用户输入的字母
    user_input = input("请输入要书写的字母（例如：JK, ABC, HELLO，空格视为空白）: ").strip()
    
    if not user_input:
        print("输入为空，请重新运行程序并输入字母。")
        return
    
    print(f"将书写字符序列: {user_input}")

    # 验证所有非空格字母都有对应的轨迹文件
    missing_letters = []
    for char in user_input:
        if char != ' ':
            csv_path = os.path.join(letters_dir, f"{char.upper()}.csv")
            if not os.path.exists(csv_path):
                missing_letters.append(char)
    
    if missing_letters:
        print(f"错误: 缺少以下字母的轨迹文件: {', '.join(missing_letters)}")
        print(f"请确保这些文件在 {letters_dir} 目录中")
        return

    # 加载MuJoCo模型
    try:
        model = mj.MjModel.from_xml_path(xml_abspath)
    except Exception as e:
        print(f"加载MuJoCo模型失败: {e}")
        print("请确保XML文件路径正确，且不包含中文字符")
        # 尝试其他可能的路径
        possible_paths = [
            xml_abspath,
            os.path.join(os.path.dirname(here), "models", "universal_robots_ur5e", "scene.xml"),
            os.path.join(os.path.dirname(os.path.dirname(here)), "models", "universal_robots_ur5e", "scene.xml"),
            "scene.xml",  # 当前目录
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"尝试使用备用路径: {path}")
                try:
                    model = mj.MjModel.from_xml_path(path)
                    print("成功加载模型")
                    break
                except:
                    continue
        else:
            raise FileNotFoundError(f"找不到可用的XML文件")
    
    data = mj.MjData(model)

    data.qpos[:] = INIT_QPOS
    mj.mj_forward(model, data)

    # ✅ 加载所有字母的轨迹段，并按照新规则添加抬笔动作
    all_segments = []
    
    # 起始点：(-0.5, 0.35)
    start_x = -0.5
    start_y = 0.35
    
    # 安全高度（抬笔高度）
    lift_height = 0.18
    
    # 当前笔的位置（从初始位置开始）
    current_pos = data.site_xpos[EE_SITE_ID].copy()
    
    # 字母计数（用于x轴偏移）
    letter_count = 0
    
    for char_index, char in enumerate(user_input):
        if char == ' ':
            # 空格处理：直接移动到下一个位置（抬笔状态）
            print(f"字符 {char_index+1}: 空格，移动到下一个位置")
            
            # 计算下一个位置的x坐标
            next_x = start_x + letter_count * 0.1
            
            # 如果当前不在安全高度，先抬笔
            if abs(current_pos[2] - lift_height) > 0.01:
                lift_segment = TrajectorySegment(
                    start_point=current_pos,
                    end_point=[current_pos[0], current_pos[1], lift_height],
                    interp_type="linear",
                    duration=0.3
                )
                all_segments.append(lift_segment)
                current_pos = [current_pos[0], current_pos[1], lift_height]
            
            # 移动到下一个位置（保持安全高度）
            move_segment = TrajectorySegment(
                start_point=current_pos,
                end_point=[next_x, start_y, lift_height],
                interp_type="linear",
                duration=0.5
            )
            all_segments.append(move_segment)
            current_pos = [next_x, start_y, lift_height]
            
            # 空格不增加字母计数，但需要移动到下一个字符位置
            letter_count += 1
            continue
        
        # 非空格字符处理
        letter = char.upper()
        csv_path = os.path.join(letters_dir, f"{letter}.csv")
        
        # 计算当前字母的x轴偏移
        x_offset = start_x + letter_count * 0.1
        
        try:
            # 加载字母的原始轨迹段
            letter_segments = load_segments_from_csv(csv_path, x_offset=x_offset, y_offset=start_y)
            print(f"加载字母 {letter} 的轨迹成功，x偏移: {x_offset:.3f}, y偏移: {start_y:.3f}, 段数: {len(letter_segments)}")
            
            if not letter_segments:
                print(f"警告: 字母 {letter} 没有轨迹段")
                letter_count += 1
                continue
            
            # 如果是第一个字符且第一个笔画的起始点不是当前位置，先移动到起始点上方
            if char_index == 0:
                first_segment_start = letter_segments[0].start_point.copy()
                
                # 先抬笔到安全高度（如果不在安全高度）
                if abs(current_pos[2] - lift_height) > 0.01:
                    lift_segment = TrajectorySegment(
                        start_point=current_pos,
                        end_point=[current_pos[0], current_pos[1], lift_height],
                        interp_type="linear",
                        duration=0.3
                    )
                    all_segments.append(lift_segment)
                    current_pos = [current_pos[0], current_pos[1], lift_height]
                
                # 移动到第一个笔画的起始点上方
                move_to_start = TrajectorySegment(
                    start_point=current_pos,
                    end_point=[first_segment_start[0], first_segment_start[1], lift_height],
                    interp_type="linear",
                    duration=0.5
                )
                all_segments.append(move_to_start)
                current_pos = [first_segment_start[0], first_segment_start[1], lift_height]
            
            # 处理字母的每一笔（每个轨迹段）
            for segment_index, segment in enumerate(letter_segments):
                segment_start = segment.start_point.copy()
                segment_end = segment.end_point.copy()
                
                # 如果不是第一笔，且当前位置不是当前笔画的起始点，先移动到起始点上方
                if segment_index > 0 or char_index > 0:
                    # 先抬笔到安全高度（如果不在安全高度）
                    if abs(current_pos[2] - lift_height) > 0.01:
                        lift_segment = TrajectorySegment(
                            start_point=current_pos,
                            end_point=[current_pos[0], current_pos[1], lift_height],
                            interp_type="linear",
                            duration=0.3
                        )
                        all_segments.append(lift_segment)
                        current_pos = [current_pos[0], current_pos[1], lift_height]
                    
                    # 移动到当前笔画的起始点上方
                    move_to_segment_start = TrajectorySegment(
                        start_point=current_pos,
                        end_point=[segment_start[0], segment_start[1], lift_height],
                        interp_type="linear",
                        duration=0.3
                    )
                    all_segments.append(move_to_segment_start)
                    current_pos = [segment_start[0], segment_start[1], lift_height]
                
                # 落笔到起始点
                lower_to_start = TrajectorySegment(
                    start_point=current_pos,
                    end_point=segment_start,
                    interp_type="linear",
                    duration=0.2
                )
                all_segments.append(lower_to_start)
                current_pos = segment_start
                
                # 添加原始轨迹段
                all_segments.append(segment)
                current_pos = segment_end
                
                # 每笔结束后都抬笔
                lift_after_segment = TrajectorySegment(
                    start_point=current_pos,
                    end_point=[current_pos[0], current_pos[1], lift_height],
                    interp_type="linear",
                    duration=0.2
                )
                all_segments.append(lift_after_segment)
                current_pos = [current_pos[0], current_pos[1], lift_height]
            
            # 字母写完后，如果后面还有字符，需要移动到下一个位置
            if char_index < len(user_input) - 1:
                next_char = user_input[char_index + 1]
                if next_char != ' ':
                    # 计算下一个字母的起始点
                    next_letter = next_char.upper()
                    next_csv_path = os.path.join(letters_dir, f"{next_letter}.csv")
                    
                    try:
                        next_segments = load_segments_from_csv(next_csv_path, 
                                                               x_offset=start_x + (letter_count + 1) * 0.1, 
                                                               y_offset=start_y)
                        if next_segments:
                            next_start = next_segments[0].start_point.copy()
                            
                            # 移动到下一个字母起始点上方
                            move_to_next = TrajectorySegment(
                                start_point=current_pos,
                                end_point=[next_start[0], next_start[1], lift_height],
                                interp_type="linear",
                                duration=0.5
                            )
                            all_segments.append(move_to_next)
                            current_pos = [next_start[0], next_start[1], lift_height]
                    except Exception as e:
                        print(f"预加载下一个字母 {next_letter} 失败: {e}")
                else:
                    # 下一个是空格，已经在上面的空格处理中处理了
                    pass
            
            letter_count += 1
                        
        except Exception as e:
            print(f"加载字母 {letter} 的轨迹失败: {e}")
            return
    
    # 在所有字符写完后，抬笔到安全高度并保持
    if all_segments:
        lift_final = TrajectorySegment(
            start_point=current_pos,
            end_point=[current_pos[0], current_pos[1], lift_height],
            interp_type="linear",
            duration=0.5
        )
        all_segments.append(lift_final)
    
    print(f"总共加载了 {len(all_segments)} 个轨迹段（包含抬笔动作）")
    
    if not all_segments:
        print("错误: 没有加载到任何轨迹段")
        return
    
    # ✅ 初始化轨迹播放器（加载所有段）
    player = TrajectoryPlayer(all_segments)
    player.reset(data.time)
    
    # ✅ 计算所有轨迹段的总时长（作为轨迹完成判断依据）
    total_all_duration = sum(seg.duration for seg in all_segments)
    print(f"所有轨迹段总时长：{total_all_duration:.2f}s，机械臂会完整执行所有段")

    glfw.init()
    window = glfw.create_window(1920, 1080, f"Writer: {user_input}", None, None)
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

    traj_points = []  # 记录书写的轨迹点（全部）
    dt_frame = 1.0 / FPS

    # 标记是否完成所有轨迹
    all_trajectory_finished = False
    # 标记是否已经完成最终姿态过渡
    final_pose_achieved = False

    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev) < dt_frame:
            # 1. 执行IK控制（所有轨迹段都执行）
            ee = data.site_xpos[EE_SITE_ID].copy()
            
            # ✅ 修改：使用DISPLAY_Z_THRESHOLD来过滤显示的轨迹点
            # 只记录z坐标小于阈值的点（即接近书写平面的点）
            if data.time >= NO_RECORD_TIME and ee[2] < DISPLAY_Z_THRESHOLD:
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

            # ✅ 检查是否完成所有轨迹段
            if not all_trajectory_finished and data.time >= total_all_duration:
                all_trajectory_finished = True
                print(f"已完成所有{len(all_segments)}段轨迹（总时长 {total_all_duration:.2f}s），开始过渡到最终姿态...")
            
            # ✅ 所有轨迹完成后，过渡到最终姿态
            if all_trajectory_finished and not final_pose_achieved:
                # 平滑过渡到最终姿态
                q_start = data.qpos.copy()
                steps = 500
                for i in range(steps):
                    alpha = (i + 1) / steps
                    q_target = (1 - alpha) * q_start + alpha * FINAL_QPOS
                    data.ctrl[:] = q_target
                    mj.mj_step(model, data)

                    # 实时渲染过渡过程
                    w, h = glfw.get_framebuffer_size(window)
                    viewport = mj.MjrRect(0, 0, w, h)
                    mj.mjv_updateScene(model, data, opt, None, cam,
                                       mj.mjtCatBit.mjCAT_ALL.value, scene)
                    
                    # ✅ 修改：显示时也使用z坐标阈值过滤
                    for p in traj_points:
                        if p[2] < DISPLAY_Z_THRESHOLD:
                            add_sphere(scene, p, TRJ_SPHERE_SIZE, TRJ_RGBA)
                    
                    mj.mjr_render(viewport, scene, context)
                    glfw.swap_buffers(window)
                    glfw.poll_events()

                    # 防止窗口关闭导致崩溃
                    if glfw.window_should_close(window):
                        break
                
                final_pose_achieved = True
                print("已到达最终姿态，可继续交互或关闭窗口")
            
            # ✅ 最终姿态达成后，保持当前控制指令
            if all_trajectory_finished and final_pose_achieved:
                data.ctrl[:] = FINAL_QPOS

            # 执行仿真步（无论是否完成轨迹都要执行，保持窗口响应）
            mj.mj_step(model, data)

        # 渲染场景
        w, h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, w, h)
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)

        # 绘制所有书写轨迹点（只显示z坐标小于阈值的点）
        for p in traj_points:
            if p[2] < DISPLAY_Z_THRESHOLD:  # ✅ 新增：使用z坐标阈值过滤
                add_sphere(scene, p, TRJ_SPHERE_SIZE, TRJ_RGBA)

        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 窗口关闭后才终止GLFW
    glfw.terminate()

    print(f"书写完成！共书写了 {len(traj_points)} 个轨迹点")
    print(f"显示阈值: z < {DISPLAY_Z_THRESHOLD}")

if __name__ == "__main__":
    main()