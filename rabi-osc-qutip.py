import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmam, mesolve

class NoisyRabi:
    def __init__(self, rabi_mhz, decay_rate):
        omega = 2*np.pi*rabi_mhz
        gamma = decay_rate #衰减率代表噪声的大小

        self.state_0 = basis(2, 0)   #激发态
        self.state_1 = basis(2, 1)   #基态

        self.H = 0.5 * omega * sigmax()

        self.c_ops = [np.sqrt(gamma) *sigmam()] #噪声衰减

        self.e_ops = [self.state_0.proj(), self.state_1.proj()]  #投影算符，定义：对任意态投影算符求期望，就是原子处于该态的概率

    def run(self, duration_us):
        times = np.linspace(0, duration_us, 200)

        result_noisy = mesolve(self.H, self.state_1, times, c_ops=self.c_ops, e_ops=self.e_ops)
        result_ideal = mesolve(self.H, self.state_1, times, c_ops=[], e_ops=self.e_ops)

        return (
            times,
            result_ideal.expect[0], result_noisy.expect[0],
            result_ideal.expect[1], result_noisy.expect[1],
        )

if __name__ ==  "__main__":
    sim = NoisyRabi(rabi_mhz=1.0, decay_rate=1.0)
    t, p1_ideal, p1_noisy, p0_ideal, p0_noisy = sim.run(duration_us=5.0)

    plt.plot(t,p1_ideal, label="(1 Ideal)", linestyle='--', color='orange')
    plt.plot(t,p1_noisy, label="(1 Noisy)", color='orange')#   横坐标时间，纵坐标概率
    plt.plot(t,p0_ideal, label="(0 Ideal)", linestyle='--', color='cyan')
    plt.plot(t,p0_noisy, label="(0 Noisy)", color='cyan')
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title('Rabi Oscillation')
    plt.legend()
    plt.show()






