import numpy as np
import matplotlib.pyplot as plt

class Qubit:
    def __init__ (self, rabi_freq_mhz, decoherence_us):
        self.omega = 2*np.pi * rabi_freq_mhz
        self.tau = decoherence_us

        self.time_axis = []
        self.state_1_prob = []
        self.state_0_prob = []

    def apply_laser_pulse(self, duration_us):
        self.time_axis = np.linspace(0, duration_us, 500)
        self.state_1_prob = 0.5 * (1 - np.exp(-self.time_axis/self.tau)*np.cos(self.omega*self.time_axis))
        self.state_0_prob = 1 - self.state_1_prob

        print(f"底层物理计算已完成： 已施加{duration_us:.3f} 微秒的激光。")

    def apply_x_gate(self):
        t_pi = np.pi / self.omega
        print(f"正在执行X门逻辑, 计算出所需的pi脉冲时间维：{t_pi:.3f}微秒")
        self.apply_laser_pulse(duration_us=t_pi)

    def plot_evolution(self):
        plt.plot(self.time_axis, self.state_1_prob, label="STATE 1 (Excited)")
        plt.plot(self.time_axis, self.state_0_prob, label="STATE 0 (Ground)")
        plt.xlabel("Time (Us)")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    print("实验开始：测试逻辑量子门")
    qubit = Qubit(rabi_freq_mhz=1.0, decoherence_us=5)
    qubit.apply_x_gate()
    qubit.plot_evolution()
