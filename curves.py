#%%
import numpy as np
import matplotlib.pyplot as plt

space = np.linspace(0, 1, 200)
p = 0.65
pwr = np.power(space, p)
plt.plot(space, pwr, label="pwr")
plt.plot(space, np.sin(space * np.pi / 2), label="sin")
k = 3
# plt.plot(space, 1 - np.exp(-k * space), label="exp basic")
k = 3
exp_norm = (1 - np.exp(-k * space)) / (1 - np.exp(-k))
plt.plot(space, exp_norm, label="exp norm")
k = 2
poly = k * space - np.power(space, k)
plt.plot(space, poly, label="poly")

plt.plot(space, (pwr + poly) / 2, label="mix")

plt.plot(space, np.log10(1 + 9*space), label="log10")

# basic_space = np.linspace(1, 100, 200)
# plt.plot(basic_space, basic_space / (basic_space + 5), label="basic norm")

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
