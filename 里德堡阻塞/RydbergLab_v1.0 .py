import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


class RydbergSimulator:
    # ==========================================
    # 第一步：构建物理世界 (定义系统的哈密顿量 H)
    # ==========================================
    def __init__(self, rabi_mhz, interaction_mhz):
        # 物理参数转化为角频率
        omega = 2 * np.pi * rabi_mhz
        V = 2 * np.pi * interaction_mhz

        # 1. 物理概念：单量子比特算符
        # Python实现：定义最基础的 2x2 NumPy 矩阵
        I = np.array([[1, 0], [0, 1]])  # 单位阵 (什么都不做)
        X = np.array([[0, 1], [1, 0]])  # 泡利 X (代表激光驱动 0 和 1 的翻转)
        N = np.array([[0, 0], [0, 1]])  # 投影算符 |1><1| (代表探测原子是否在里德堡态)

        # 2. 物理概念：构建双比特系统的哈密顿量 (张量积)
        # Python实现：使用 np.kron() 计算两个矩阵的张量积，生成 4x4 矩阵

        # 驱动项 H_drive：激光同时打在原子1和原子2上
        H_drive = 0.5 * omega * (np.kron(X, I) + np.kron(I, X))

        # 相互作用项 H_int：当且仅当两个原子都在 |1> 态时，产生巨大的能量惩罚 V
        H_int = V * np.kron(N, N)

        # 系统的总哈密顿量
        self.H = H_drive + H_int

        # 打印4×4矩阵
        print(f"\n当前系统的哈密顿量矩阵 （V={interaction_mhz}):")
        print(np.round(self.H / (2 * np.pi), 2))  # 除以2pi方便看MHz

        # 准备空列表，用于记录随时间变化的概率
        self.times = []
        self.prob_00 = []  # 基态概率
        self.prob_W = []  # 纠缠态 (|01> + |10>) 概率
        self.prob_11 = []  # 双激发态概率

    # ==========================================
    # 第二步：启动物理引擎 (求解含时薛定谔方程)
    # ==========================================
    def run_evolution(self, duration_us):
        self.times = np.linspace(0, duration_us, 200)

        # 物理概念：系统的初始状态是两个原子都在基态 |00>
        # Python实现：定义一个 4x1 的列向量
        psi_0 = np.array([1, 0, 0, 0], dtype=complex)

        for t in self.times:
            # 物理概念：时间演化算符 U = exp(-iHt)
            # Python实现：使用 scipy.linalg.expm 计算矩阵的指数
            U = la.expm(-1j * self.H * t)

            # 物理概念：状态随时间演化 |psi(t)> = U |psi(0)>
            # Python实现：使用 @ 符号进行矩阵和向量的乘法
            psi_t = U @ psi_0

            # 物理概念：玻恩法则，概率等于复数振幅绝对值的平方
            probs = np.abs(psi_t) ** 2

            # 提取 4 个状态的概率存入列表
            self.prob_00.append(probs[0])  # 对应 |00>
            self.prob_W.append(probs[1] + probs[2])  # 对应 |01> 和 |10> 的和
            self.prob_11.append(probs[3])  # 对应 |11>

    # ==========================================
    # 第三步：极简结果展示 (剥离所有花哨排版)
    # ==========================================
    def show_results(self, title):
        plt.plot(self.times, self.prob_00, label="State |00>")
        plt.plot(self.times, self.prob_W, label="State |01> + |10>")
        plt.plot(self.times, self.prob_11, label="State |11>")
        plt.title(title)
        plt.legend()
        plt.show()


# ==========================================
# 第四步：主程序 (对比“有无里德堡阻塞”的物理现象)
# ==========================================
if __name__ == "__main__":
    # 实验 A：原子相距很远，V = 0 (无阻塞，独立演化)
    sim_far = RydbergSimulator(rabi_mhz=1.0, interaction_mhz=0.0)
    sim_far.run_evolution(duration_us=3.0)
    sim_far.show_results("Experiment A: No Blockade (V = 0)")

    # 实验 B：原子靠得很近，V = 50 (产生强烈的里德堡阻塞！)
    sim_close = RydbergSimulator(rabi_mhz=1.0, interaction_mhz=50.0)
    sim_close.run_evolution(duration_us=3.0)
    sim_close.show_results("Experiment B: Rydberg Blockade (V = 50)")
