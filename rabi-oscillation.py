import numpy as np
import matplotlib.pyplot as plt
#定义量子比特
class Qubit:
    def __init__(self, rabi_freq_mhz, decoherence_us):  #定义频率寿命
        self.omega = 2*np.pi*rabi_freq_mhz
        self.tau = decoherence_us
        #三个表格
        self.time_axis = []
        self.state_1_prob = []
        self.state_0_prob = []
        #定义打激光时长
    def apply_laser_pulse(self, duration_us):
            self.time_axis = np.linspace(0, duration_us, 500)
            self.state_1_prob = 0.5*(1-np.exp(-self.time_axis / self.tau)*np.cos(self.omega*self.time_axis))
            self.state_0_prob = 1-self.state_1_prob

            print("计算完成")
    #画图部分
    def plot_evolution(self):
            plt.plot(self.time_axis, self.state_1_prob, label="STATE 1")
            plt.plot(self.time_axis, self.state_0_prob, label="STATE 0")
            plt.legend()
            plt.show()
#主程序运行，可改参数
if __name__ == "__main__":
    print("实验开始")
    qubit_1 = Qubit(rabi_freq_mhz=1.0, decoherence_us=3.0)#频率影响周期，寿命影响退相干
    qubit_2 = Qubit(rabi_freq_mhz=1.0, decoherence_us=20.0)
    qubit_3 = Qubit(rabi_freq_mhz=1.0, decoherence_us=30.0)

    print("测试第一个量子比特")
    qubit_1.apply_laser_pulse(duration_us=10)#打激光时长
    qubit_1.plot_evolution()
    print("测试第二个量子比特")
    qubit_2.apply_laser_pulse(duration_us=10)
    qubit_2.plot_evolution()
    print("测试第三个量子比特")
    qubit_3.apply_laser_pulse(duration_us=10)
    qubit_3.plot_evolution()
