from scipy.stats import describe
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
values = np.random.beta(a=2.5, b=30.0, size=10) * 10  # scale to 0–10 range


plt.hist(values, bins=100)
plt.savefig('./hst.png')
plt.close()

values = np.zeros((1000, 2))

for dist in range(2):
    values[:,dist] = np.random.beta(a=2.5, b=30.0, size=1000) * 10  # scale to 0–10 range
    plt.hist(values[:,dist], bins=100)
    plt.savefig(f'./hst{dist}.png')
    plt.close()


print(describe(values, nan_policy='omit'))