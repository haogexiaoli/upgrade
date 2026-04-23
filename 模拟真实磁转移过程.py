import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#绘图设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman',    'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = ('stix')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#导入数据
df = pd.read_csv('real_lab_data.csv')
x_data = df['Position_mm'].values


#定义高斯公式，告诉电脑物理规律
def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2/ (2*sigma ** 2))

#绘图与自动拟合
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 0.9, 5))   #0和0.9什么意思

for i in range(5):
    y_exp = df[f'OD_t{i}'].values

    #画出散点
    ax.plot(x_data[::15], y_exp[::15], 'o', color=colors[i], markersize=5,
            alpha=0.4, markeredgewidth=0)     #代码的意义
    guess = [1.0, i * 2.5, 0.5]   #怎样猜测的？
    popt, pcov = curve_fit(gaussian, x_data, y_exp, p0=guess)
    best_A, best_x0, best_sigma = popt
    y_fit = gaussian(x_data, best_A, best_x0, best_sigma)
    ax.plot(x_data, y_fit, '-', color=colors[i], linewidth=2.5,
            label=f'$t_{i}$: $x_0$={best_x0:.1f}mm')   #label

#美化图例
ax.set_xlabel('Position $x$(mm)', fontsize=14)
ax.set_ylabel('Optical Density(OD)', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(frameon=False, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()


