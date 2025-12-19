# EE_class_project
A dustbin for the project. Upload your update data and file here.

咱是robotwriter
————————————————————————————————————————————————————————————————————————————
提取出的逆运动学函数
import numpy as np
import scipy as sp
import mujoco as mj

def inverse_kinematics(model, data, target_pos, target_quat=None, target_euler=None, 
                       max_iterations=100, tolerance=1e-4, damping=0.01, Kp=0.1):
    """
    使用雅可比伪逆法求解逆运动学
    
    参数:
    ----------
    model : mujoco.MjModel
        MuJoCo模型
    data : mujoco.MjData
        MuJoCo数据
    target_pos : np.ndarray (3,)
        目标位置 [x, y, z]
    target_quat : np.ndarray (4,), 可选
        目标姿态（四元数 [w, x, y, z]）
    target_euler : np.ndarray (3,), 可选
        目标姿态（欧拉角 [phi, theta, psi]，XYZ顺序）
    max_iterations : int
        最大迭代次数
    tolerance : float
        收敛容差
    damping : float
        阻尼系数（防止奇异）
    Kp : float
        比例增益
        
    返回:
    -------
    q_solution : np.ndarray
        关节角度解
    success : bool
        是否成功收敛
    """
    
    # 当前关节角度
    q_current = data.qpos.copy()
    
    for iteration in range(max_iterations):
        # 1. 设置当前关节角度并计算正向运动学
        data.qpos[:] = q_current
        mj.mj_forward(model, data)
        
        # 2. 获取当前末端位姿
        current_pos = data.site_xpos[0].copy()  # 位置
        current_mat = data.site_xmat[0].copy()  # 旋转矩阵
        
        # 3. 计算位置误差
        pos_error = target_pos - current_pos
        
        # 4. 计算姿态误差
        if target_quat is not None:
            # 使用四元数表示姿态
            current_quat = np.zeros(4)
            mj.mju_mat2Quat(current_quat, current_mat)
            
            # 四元数误差：q_error = q_target * q_current^-1
            q_current_inv = np.array([current_quat[0], -current_quat[1], 
                                      -current_quat[2], -current_quat[3]])
            q_error = quat_multiply(target_quat, q_current_inv)
            
            # 转换为轴角表示（用于雅可比矩阵）
            angle = 2 * np.arccos(np.clip(q_error[0], -1, 1))
            if angle > np.pi:
                angle = 2 * np.pi - angle
            
            if angle < 1e-6:
                axis = np.array([1, 0, 0])
            else:
                axis = q_error[1:] / np.sqrt(1 - q_error[0]**2)
            
            rot_error = axis * angle
            
        elif target_euler is not None:
            # 使用欧拉角表示姿态
            current_quat = np.zeros(4)
            mj.mju_mat2Quat(current_quat, current_mat)
            r_current = sp.spatial.transform.Rotation.from_quat(
                [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
            )
            current_euler = r_current.as_euler('xyz')
            rot_error = target_euler - current_euler
        else:
            # 如果没有指定姿态，只控制位置
            rot_error = np.zeros(3)
        
        # 5. 计算总误差
        error = np.concatenate([pos_error, rot_error])
        
        # 检查是否收敛
        if np.linalg.norm(error) < tolerance:
            return q_current, True
        
        # 6. 计算雅可比矩阵
        jacp = np.zeros((3, model.nq))
        jacr = np.zeros((3, model.nq))
        mj.mj_jac(model, data, jacp, jacr, current_pos, 7)  # 7是末端执行器的body ID
        
        J = np.vstack((jacp, jacr))
        
        # 7. 计算阻尼伪逆
        n_joints = J.shape[1]
        Jinv = np.linalg.pinv(J.T @ J + damping**2 * np.eye(n_joints)) @ J.T
        
        # 8. 计算关节角度增量
        dq = Kp * Jinv @ error
        
        # 9. 更新关节角度
        q_current += dq
        
        # 10. 施加关节限位
        for i in range(model.nq):
            q_current[i] = np.clip(q_current[i], 
                                   model.jnt_range[i, 0] if model.jnt_range[i, 0] < model.jnt_range[i, 1] else -np.pi,
                                   model.jnt_range[i, 1] if model.jnt_range[i, 0] < model.jnt_range[i, 1] else np.pi)
    
    # 达到最大迭代次数仍未收敛
    return q_current, False

def quat_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])



逆运动学函数调用方法
# 加载模型
xml_path = '../../models/universal_robots_ur5e/scene.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 定义目标位姿
target_position = np.array([0.5, 0.2, 0.3])  # [x, y, z]
target_euler = np.array([3.14, 0, 0])  # [phi, theta, psi] XYZ欧拉角

# 从home位置开始
data.qpos[:] = model.key("home").qpos.copy()

# 计算逆运动学
joint_angles, success = inverse_kinematics(
    model, data, 
    target_pos=target_position,
    target_euler=target_euler,
    max_iterations=100,
    tolerance=1e-4,
    damping=0.01,
    Kp=0.1
)

if success:
    print("逆运动学求解成功！")
    print(f"关节角度: {joint_angles}")
else:
    print("逆运动学求解失败，可能目标不可达")
