
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 1. 改变质量（最常用）
#model.body_mass[1] *= 1.5

# 2. 施加外力（最直接）
#data.xfrc_applied[1][0] = 2.0  # x方向2N

# 3. 改变角度（最快速）
#data.qpos[0] += 0.5  # 偏离0.5弧度

# 4. 改变速度（模拟冲击）
#data.qvel[0] = 1.0   # 1rad/s的初速度

xml_path = '../../models/Pendulum/pendulum.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)

# =================================================================
# 1. 直接修改物理参数（最根本的物理模型扰动）
# =================================================================
class PhysicalParameterDisturbance:
    """通过修改模型物理参数实现扰动"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.original_params = {}
        self.save_original_params()
        
    def save_original_params(self):
        """保存原始物理参数"""
        # 保存几何参数
        self.original_params['geom_size'] = self.model.geom_size.copy()
        self.original_params['geom_pos'] = self.model.geom_pos.copy()
        
        # 保存惯性参数
        self.original_params['body_mass'] = self.model.body_mass.copy()
        self.original_params['body_inertia'] = self.model.body_inertia.copy()
        
        # 保存关节参数
        self.original_params['joint_stiffness'] = self.model.jnt_stiffness.copy()
        self.original_params['damping'] = self.model.dof_damping.copy()
        
        # 保存重力
        self.original_params['gravity'] = self.model.opt.gravity.copy()
    
    def apply_mass_change(self, body_id, factor):
        """改变质量：模拟负载变化"""
        original_mass = self.original_params['body_mass'][body_id]
        self.model.body_mass[body_id] = original_mass * factor
        print(f"Body {body_id} mass changed by factor {factor}")
        
    def apply_length_change(self, geom_id, factor):
        """改变摆杆长度：模拟结构变化"""
        original_size = self.original_params['geom_size'][geom_id]
        self.model.geom_size[geom_id] = original_size * factor
        print(f"Geometry {geom_id} size changed by factor {factor}")
        
    def apply_gravity_change(self, factor):
        """改变重力：模拟不同重力环境"""
        original_gravity = self.original_params['gravity']
        self.model.opt.gravity = original_gravity * factor
        print(f"Gravity changed by factor {factor}")
        
    def apply_damping_change(self, factor):
        """改变阻尼：模拟润滑/摩擦变化"""
        original_damping = self.original_params['damping']
        self.model.dof_damping = original_damping * factor
        print(f"Damping changed by factor {factor}")
    
    def restore_original_params(self):
        """恢复原始物理参数"""
        for param, value in self.original_params.items():
            if hasattr(self.model, param):
                setattr(self.model, param, value.copy())
        print("Original physical parameters restored")

# =================================================================
# 2. 直接施加外力（在物理层面）
# =================================================================
class ExternalForceDisturbance:
    """通过施加外力实现扰动"""
    
    def __init__(self, model, data, body_name="pendulum"):
        self.model = model
        self.data = data
        # 获取物体ID
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            print(f"Body '{body_name}' not found, using body 0")
            self.body_id = 0
    
    def apply_impulse_force(self, force_vector, point=None):
        """
        施加脉冲力
        force_vector: [fx, fy, fz] 在世界坐标系中的力
        point: 施加点的局部坐标（None表示在质心）
        """
        if point is None:
            # 在质心施加力
            self.data.xfrc_applied[self.body_id][:3] = force_vector
            self.data.xfrc_applied[self.body_id][3:] = 0  # 无扭矩
        else:
            # 在指定点施加力和产生的扭矩
            # r × F 计算扭矩
            torque = np.cross(point, force_vector)
            self.data.xfrc_applied[self.body_id][:3] = force_vector
            self.data.xfrc_applied[self.body_id][3:] = torque
    
    def apply_constant_force(self, force_vector, duration):
        """施加恒定力"""
        # 在实际实现中，需要在多个时间步中保持力的施加
        pass
    
    def clear_force(self):
        """清除所有外力"""
        self.data.xfrc_applied[self.body_id][:] = 0.0

# =================================================================
# 3. 修改状态直接扰动
# =================================================================
class StateDisturbance:
    """通过直接修改系统状态实现扰动"""
    
    def __init__(self, data):
        self.data = data
        self.original_qpos = None
        self.original_qvel = None
    
    def save_current_state(self):
        """保存当前状态"""
        self.original_qpos = self.data.qpos.copy()
        self.original_qvel = self.data.qvel.copy()
    
    def apply_position_disturbance(self, angle_offset):
        """施加角度扰动"""
        self.data.qpos[0] += angle_offset
        print(f"Applied position disturbance: +{angle_offset} rad")
    
    def apply_velocity_disturbance(self, velocity_offset):
        """施加速度扰动"""
        self.data.qvel[0] += velocity_offset
        print(f"Applied velocity disturbance: +{velocity_offset} rad/s")
    
    def apply_impulse_velocity(self, impulse):
        """施加速度脉冲（模拟冲击）"""
        self.data.qvel[0] = impulse
        print(f"Applied velocity impulse: {impulse} rad/s")
    
    def restore_state(self):
        """恢复保存的状态"""
        if self.original_qpos is not None:
            self.data.qpos[:] = self.original_qpos
        if self.original_qvel is not None:
            self.data.qvel[:] = self.original_qvel
        print("State restored")

# =================================================================
# 4. 环境扰动
# =================================================================
class EnvironmentalDisturbance:
    """模拟环境变化扰动"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.original_wind = np.zeros(3)
        
    def apply_wind(self, wind_velocity):
        """
        施加风力扰动
        简化模型：风力与相对速度的平方成正比
        """
        # 获取摆杆末端速度
        body_vel = self.data.qvel[0] * 0.5  # 假设摆杆长度为0.5m
        
        # 计算相对风速
        relative_wind = wind_velocity - body_vel
        
        # 计算风阻力（简化模型）
        drag_coefficient = 0.5
        area = 0.01  # 迎风面积
        air_density = 1.225
        
        drag_force = 0.5 * air_density * drag_coefficient * area * relative_wind**2
        drag_force *= np.sign(relative_wind)  # 方向与相对风速相反
        
        # 施加力
        self.data.xfrc_applied[0][0] = drag_force
        
        return drag_force

# =================================================================
# PID控制器
# =================================================================
def pid(kp, ki, kd, state, dt):
    current_error = state['error']
    state['integral'] += current_error * dt
    derivative = (current_error - state['last_error']) / dt
    output = (kp * current_error) + (ki * state['integral']) + (kd * derivative)
    state['last_error'] = current_error
    return output

# =================================================================
# 主程序：演示各种物理模型扰动
# =================================================================
def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path(abspath)
    data = mujoco.MjData(model)
    
    # 控制器参数
    KP = 60.0
    KI = 2.0
    KD = 8.0
    TARGET_ANGLE = np.pi
    DT = model.opt.timestep
    
    # 初始化扰动管理器
    param_disturbance = PhysicalParameterDisturbance(model, data)
    force_disturbance = ExternalForceDisturbance(model, data, "pendulum")
    state_disturbance = StateDisturbance(data)
    env_disturbance = EnvironmentalDisturbance(model, data)
    
    # 控制状态
    control_state = {
        'error': 0.0,
        'integral': 0.0,
        'last_error': 0.0
    }
    
    # 数据记录
    time_history = []
    angle_history = []
    error_history = []
    torque_history = []
    disturbance_type_history = []
    disturbance_magnitude_history = []
    
    # 扰动序列
    disturbances = [
        {"time": 2.0, "type": "mass_change", "magnitude": 2.0, "duration": 1.0},
        {"time": 4.0, "type": "external_force", "magnitude": 3.0, "duration": 0.5},
        {"time": 6.0, "type": "state_velocity", "magnitude": 1.5, "duration": 0.1},
        {"time": 8.0, "type": "gravity_change", "magnitude": 0.5, "duration": 2.0},
    ]
    
    current_disturbance = None
    disturbance_start_time = None
    
    print(f"Loading model from: {abspath}")
    print(f"Disturbances scheduled: {disturbances}")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_sim_time = data.time
        
        while viewer.is_running() and data.time - start_sim_time < 12.0:
            step_start = time.time()
            
            current_time = data.time
            
            # 检查是否需要开始新的扰动
            for dist in disturbances:
                if (dist["time"] <= current_time < dist["time"] + dist["duration"] and 
                    current_disturbance != dist):
                    current_disturbance = dist
                    disturbance_start_time = current_time
                    apply_disturbance(dist, param_disturbance, force_disturbance, 
                                    state_disturbance, env_disturbance, data)
            
            # 检查是否需要结束当前扰动
            if (current_disturbance and 
                current_time >= disturbance_start_time + current_disturbance["duration"]):
                remove_disturbance(current_disturbance, param_disturbance, force_disturbance)
                current_disturbance = None
            
            # 持续的环境扰动（如风）
            if 5.0 <= current_time < 7.0:
                wind_force = env_disturbance.apply_wind(np.array([1.0, 0, 0]))
                disturbance_magnitude_history.append(wind_force)
                disturbance_type_history.append("wind")
            else:
                env_disturbance.apply_wind(np.array([0, 0, 0]))
                disturbance_magnitude_history.append(0)
                disturbance_type_history.append("none")
            
            # PID控制
            control_state['error'] = TARGET_ANGLE - data.qpos[0]
            torque = pid(KP, KI, KD, control_state, DT)
            data.ctrl[0] = torque
            
            # 记录数据
            time_history.append(current_time)
            angle_history.append(data.qpos[0])
            error_history.append(control_state['error'])
            torque_history.append(torque)
            
            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 实时同步
            elapsed = time.time() - step_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
        
        # 恢复原始参数
        param_disturbance.restore_original_params()
    
    # 绘图
    plot_results(time_history, angle_history, error_history, torque_history,
                disturbance_type_history, disturbance_magnitude_history)

def apply_disturbance(disturbance, param_dist, force_dist, state_dist, env_dist, data):
    """应用指定的物理扰动"""
    dist_type = disturbance["type"]
    magnitude = disturbance["magnitude"]
    
    print(f"\nApplying {dist_type} disturbance at t={data.time:.2f}s, magnitude={magnitude}")
    
    if dist_type == "mass_change":
        # 改变质量：模拟突然增加负载
        param_dist.apply_mass_change(1, magnitude)
        
    elif dist_type == "external_force":
        # 施加外部力：模拟推力或拉力
        force_vector = np.array([magnitude, 0, 0])  # x方向
        force_dist.apply_impulse_force(force_vector)
        
    elif dist_type == "state_velocity":
        # 直接修改速度：模拟冲击
        state_dist.save_current_state()
        state_dist.apply_impulse_velocity(magnitude)
        
    elif dist_type == "gravity_change":
        # 改变重力：模拟不同环境
        param_dist.apply_gravity_change(magnitude)
        
    elif dist_type == "damping_change":
        # 改变阻尼：模拟润滑变化
        param_dist.apply_damping_change(magnitude)
        
    elif dist_type == "length_change":
        # 改变长度：模拟结构变形
        param_dist.apply_length_change(0, magnitude)
    
    disturbance_type_history.append(dist_type)
    disturbance_magnitude_history.append(magnitude)

def remove_disturbance(disturbance, param_dist, force_dist):
    """移除扰动"""
    dist_type = disturbance["type"]
    
    if dist_type == "external_force":
        force_dist.clear_force()
    elif dist_type == "gravity_change":
        param_dist.restore_original_params()
    elif dist_type == "mass_change":
        param_dist.restore_original_params()
    elif dist_type == "damping_change":
        param_dist.restore_original_params()
    elif dist_type == "length_change":
        param_dist.restore_original_params()
    
    print(f"Removed {dist_type} disturbance")

def plot_results(time_history, angle_history, error_history, torque_history,
                disturbance_type_history, disturbance_magnitude_history):
    """绘制结果"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # 1. 角度跟踪
    axes[0, 0].plot(time_history, angle_history, 'b-', linewidth=2)
    axes[0, 0].axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].set_title('Pendulum Angle Response')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. 控制误差
    axes[0, 1].plot(time_history, error_history, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (rad)')
    axes[0, 1].set_title('Control Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 控制扭矩
    axes[1, 0].plot(time_history, torque_history, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].set_title('Control Torque')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 扰动时间线
    axes[1, 1].scatter(time_history[:len(disturbance_type_history)], 
                      [1] * len(disturbance_type_history), 
                      c='red', s=50, alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title('Disturbance Timeline')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 扰动幅度
    axes[2, 0].plot(time_history[:len(disturbance_magnitude_history)], 
                   disturbance_magnitude_history, 'orange', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Disturbance Magnitude')
    axes[2, 0].set_title('Disturbance Magnitude')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 性能统计
    axes[2, 1].axis('off')
    stats_text = f"""
    PERFORMANCE SUMMARY
    
    Simulation Duration: {time_history[-1]:.2f}s
    
    Error Statistics:
    - Max Error: {np.max(np.abs(error_history)):.4f} rad
    - RMS Error: {np.sqrt(np.mean(np.square(error_history))):.6f} rad
    - Steady-state Error: {np.mean(np.abs(error_history[-100:])):.6f} rad
    
    Torque Statistics:
    - Max Torque: {np.max(np.abs(torque_history)):.3f} Nm
    - Avg Torque: {np.mean(np.abs(torque_history)):.3f} Nm
    """
    axes[2, 1].text(0.05, 0.5, stats_text, fontsize=10, 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('physical_disturbance_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# =================================================================
# 专门测试特定物理扰动的函数
# =================================================================
def test_specific_physical_disturbance(disturbance_type, magnitude=2.0):
    """测试特定的物理扰动"""
    model = mujoco.MjModel.from_xml_path(abspath)
    data = mujoco.MjData(model)
    
    # 控制器参数
    KP = 60.0
    KI = 2.0
    KD = 8.0
    TARGET_ANGLE = np.pi
    DT = model.opt.timestep
    
    # 初始化扰动管理器
    param_disturbance = PhysicalParameterDisturbance(model, data)
    force_disturbance = ExternalForceDisturbance(model, data)
    state_disturbance = StateDisturbance(data)
    
    # 控制状态
    control_state = {
        'error': 0.0,
        'integral': 0.0,
        'last_error': 0.0
    }
    
    time_history = []
    angle_history = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = data.time
        
        while viewer.is_running() and data.time - start_time < 8.0:
            current_time = data.time
            
            # 在4秒时施加扰动
            if 4.0 <= current_time < 6.0:
                if disturbance_type == "mass":
                    param_disturbance.apply_mass_change(1, magnitude)
                elif disturbance_type == "gravity":
                    param_disturbance.apply_gravity_change(magnitude)
                elif disturbance_type == "force":
                    force_disturbance.apply_impulse_force([magnitude, 0, 0])
                elif disturbance_type == "velocity":
                    if current_time < 4.1:  # 只在开始时施加一次
                        state_disturbance.apply_impulse_velocity(magnitude)
            
            # 控制
            control_state['error'] = TARGET_ANGLE - data.qpos[0]
            torque = pid(KP, KI, KD, control_state, DT)
            data.ctrl[0] = torque
            
            # 记录
            time_history.append(current_time)
            angle_history.append(data.qpos[0])
            
            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()
        
        # 恢复
        param_disturbance.restore_original_params()
        force_disturbance.clear_force()
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, angle_history, 'b-', linewidth=2)
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Target')
    plt.axvspan(4.0, 6.0, alpha=0.2, color='gray', label=f'{disturbance_type} disturbance')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Physical Disturbance Test: {disturbance_type} (magnitude={magnitude})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{disturbance_type}_disturbance_test.png', dpi=150)
    plt.show()

# =================================================================
# 运行主程序或特定测试
# =================================================================
if __name__ == "__main__":
    print("Physical Disturbance Test for Pendulum System")
    print("=" * 50)
    print("Options:")
    print("1. Run full disturbance sequence")
    print("2. Test mass disturbance")
    print("3. Test gravity disturbance")
    print("4. Test external force disturbance")
    print("5. Test velocity impulse disturbance")
    
    choice = input("\nSelect option (1-5): ").strip()
    

    #修改此处参数以调整扰动程度
    if choice == "1":
        main()
    elif choice == "2":
        test_specific_physical_disturbance("mass", magnitude=2.0)
    elif choice == "3":
        test_specific_physical_disturbance("gravity", magnitude=5)
    elif choice == "4":
        test_specific_physical_disturbance("force", magnitude=3)
    elif choice == "5":
        test_specific_physical_disturbance("velocity", magnitude=2.0)
    else:
        print("Invalid choice. Running full sequence...")
        main()