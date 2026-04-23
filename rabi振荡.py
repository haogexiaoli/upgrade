import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 依然保持我们引以为傲的科研级排版
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# 2. 设定量子物理参数
# ==========================================
Omega = 2 * np.pi * 1.0  # 拉比频率 (Rabi Frequency)：决定振荡有多快 (这里设为 1 MHz)
tau = 3.0                # 退相干时间 (Decoherence time)：决定振荡能维持多久 (微秒)

# 时间轴：从 0 到 5 微秒
t = np.linspace(0, 5, 500)

# ==========================================
# 3. 物理公式：计算处于状态 0 和状态 1 的概率
# ==========================================
# 处于激发态 |1> 的概率 (带有指数衰减)
P1 = 0.5 * (1 - np.exp(-t / tau) * np.cos(Omega * t))

# 处于基态 |0> 的概率 (总概率为 1)
P0 = 1 - P1

# ==========================================
# 4. 开始绘图
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# 画出状态 0 和 1 的概率曲线
ax.plot(t, P1, '-', color='#D62728', linewidth=2.5, label=r'Excited State $|1\rangle$')
ax.plot(t, P0, '-', color='#1F77B4', linewidth=2.5, label=r'Ground State $|0\rangle$')

# ==========================================
# 5. 标注量子计算的核心：Pi 脉冲 和 Pi/2 脉冲
# ==========================================
t_pi = np.pi / Omega      # Pi 脉冲的时间 (刚好翻转到 |1>)
t_pi_half = t_pi / 2      # Pi/2 脉冲的时间 (达到 50/50 叠加态)

# 画垂直虚线指示这两个关键时间点
ax.axvline(t_pi, color='gray', linestyle='--', alpha=0.7)
ax.axvline(t_pi_half, color='gray', linestyle='--', alpha=0.7)

# 添加文字标注
ax.text(t_pi, 1.05, r'$\pi$-pulse (NOT Gate)', ha='center', fontsize=12, color='black')
ax.text(t_pi_half, 1.05, r'$\pi/2$-pulse', ha='center', fontsize=12, color='black')

# ==========================================
# 6. 美化图表
# ==========================================
ax.set_xlabel(r'Laser Pulse Duration $t$ ($\mu$s)', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
ax.set_ylim(-0.05, 1.15) # Y轴稍微留点空隙，放得下文字
ax.tick_params(axis='both', labelsize=12)
ax.legend(frameon=False, fontsize=12, loc='center right')

plt.tight_layout()
plt.show()
