import numpy as np
import logging
from dataclasses import dataclass #规范数据结构
from typing import List, Tuple, Dict, Union #类型提示
import matplotlib.pyplot as plt
from qutip import basis, sigmam, sigmax, qeye, tensor, mesolve, Qobj

#配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S")

logger = logging.getLogger("RydbergSimulator")  #创建一个记录器
@dataclass
class RydbergConfig:
    rabi_mhz: float     #拉比频率
    blockade_v_mhz:float   #相互作用
    decay_rate:float  #噪声    退相干的作用
    pulse_width_us: float #激光开启时间

def laser_pulse(t, args):
    pulse_width = args.get('pulse_width', 20.0)
    if t <=pulse_width:
        return 1.0
    else:
        return 0.0

class RydbergSimulator:
    def __init__(self, config: RydbergConfig):
        self.config = config
        logger.info(f"初始化仿真器 （Rabi={config.rabi_mhz}MHz, V={config.blockade_v_mhz}MHz， decay={config.decay_rate}MHz")
        self.H: Union[Qobj, list] = None
        self.initial_state: Qobj = None  #None是什么意思
        self.e_ops: List[Qobj] = []
        self.c_ops: List[Qobj] = []

        self.state_00: Qobj = None
        self.state_01: Qobj = None
        self.state_10: Qobj = None
        self.state_11: Qobj = None

        self._define_quantum_states()
        self._build_hamiltonian()
        self._build_collapse_operators()
        self._setup_measurements()
        logger.info("物理系统构建完毕（态、 哈密顿量、 干扰、 测量仪已就绪。")


    def _define_quantum_states(self) -> None:
        logger.debug("正在定义量子态")
        s0, s1 = basis(2,1), basis(2,0)
        self.state_00 = tensor(s0, s0)
        self.state_01 = tensor(s0, s1)
        self.state_10 = tensor(s1, s0)
        self.state_11 = tensor(s1, s1)

        self.initial_state = self.state_00

    def _build_hamiltonian(self) -> None:
        logger.debug("正在构建哈密顿量")
        omega = 2 * np.pi *self.config.rabi_mhz
        V = 2 * np.pi *self.config.blockade_v_mhz
        sx, id_op = sigmax(), qeye(2)
        n_op = basis(2, 0).proj()
        h_drive1 = 0.5 * omega * tensor(sx, id_op)
        h_drive2 = 0.5 * omega * tensor(id_op, sx)
        h_driver = h_drive1 + h_drive2
        h_int = V * tensor(n_op, n_op)

        self.H = [h_int, [h_driver, laser_pulse]]

    def _setup_measurements(self) -> None:
        logger.debug("正在设置期望值测量算符")
        self.e_ops = [
            self.state_00.proj(),
            self.state_01.proj(),
            self.state_10.proj(),
            self.state_11.proj(),]

    def _build_collapse_operators(self) -> None:
        gamma = self.config.decay_rate
        if gamma > 0.0:
            logger.debug("正在加入退相干耗散")
            sm = sigmam()
            id_op = qeye(2)
            c1 = np.sqrt(gamma) * tensor(sm, id_op)
            c2 = np.sqrt(gamma) * tensor(id_op, sm)
            self.c_ops = [c1, c2]


    def simulate(self, duration_us: float, steps: int = 1000) ->Tuple[np.ndarray, Dict[str, np.ndarray]]:
        logger.info(f"开始执行量子演化仿真，时长：{duration_us} us")
        times = np.linspace(0, duration_us, steps)
        pulse_args = {'pulse_width': self.config.pulse_width_us}
        result = mesolve(self.H, self.initial_state, times, c_ops=self.c_ops, e_ops=self.e_ops, args=pulse_args)

        results_dict = {
            "00":result.expect[0],
            "01":result.expect[1],
            "10":result.expect[2],
            "11":result.expect[3],}

        return times, results_dict

#画图部分
def plot_dynamics(times:np.ndarray, results:Dict[str, np.ndarray], title: str) -> None:
    logger.info("正在生成图表")
    plt.figure(figsize=(9, 5))
    plt.plot(times, results["00"],label="State|00>")
    plt.plot(times, results["01"] + results["10"],label="State|01> + |10>")
    plt.plot(times, results["11"] ,label="State|11>", color='red',linewidth=4)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


#主程序
if __name__ == "__main__":
    config = RydbergConfig(rabi_mhz=1.0, blockade_v_mhz=50.0, decay_rate=0.2,  pulse_width_us=5.0)
    simulator = RydbergSimulator(config)
    t, prob_data = simulator.simulate(duration_us=20.0)
    plot_dynamics(t, prob_data, f"Rydberg Dynamics(V = {config.blockade_v_mhz} MHz, Pulse={config.pulse_width_us} us)")
