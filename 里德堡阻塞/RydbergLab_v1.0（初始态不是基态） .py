import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


class RydbergSimulator:
    def __init__(self, rabi_mhz, interaction_mhz):
        omega = 2 * np.pi * rabi_mhz
        V = 2 * np.pi * interaction_mhz

        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        N = np.array([[0, 0], [0, 1]])

        H_drive = 0.5 * omega * (np.kron(X, I) + np.kron(I, X))
        H_int = V * np.kron(N, N)
        self.H = H_drive + H_int

        self.times = []
        self.prob_00 = []
        self.prob_01 = []
        self.prob_10 = []
        self.prob_11 = []

    def run_evolution(self, duration_us):
        self.times = np.linspace(0, duration_us, 200)

        psi_0 = np.array([0, 0, 1, 0], dtype=complex)

        for t in self.times:
            U = la.expm(-1j * self.H * t)
            psi_t = U @ psi_0
            probs = np.abs(psi_t) ** 2

            self.prob_00.append(probs[0])
            self.prob_01.append(probs[1])
            self.prob_10.append(probs[2])
            self.prob_11.append(probs[3])

    def show_results(self, title):
        plt.figure(figsize=(8, 5))
        plt.plot(self.times, self.prob_00, label="State |00> (Bridge)", linestyle='--')
        plt.plot(self.times, self.prob_01, label="State |01> (Atom 2 Excited)", linewidth=2)
        plt.plot(self.times, self.prob_10, label="State |10> (Atom 1 Excited)", linewidth=2)
        plt.plot(self.times, self.prob_11, label="State |11> (Blocked)", color='red')

        plt.title(title)
        plt.xlabel("Time (us)")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    # 运行强阻塞实验，初始态为 |10>
    sim = RydbergSimulator(rabi_mhz=1.0, interaction_mhz=50.0)
    sim.run_evolution(duration_us=4.0)
    sim.show_results("Excitation Exchange: |10> <--> |01>")
