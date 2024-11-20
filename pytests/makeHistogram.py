import numpy as np
import math
import matplotlib.pyplot as plt
import json


reduction = open ('./jsonfiles/random_distributed_reduction_factorNO_ID.json')
density = open ('./jsonfiles/random_distributed_numb_densityNO_ID.json')

reduction_data = json.load(reduction)
density_data = json.load(density)
log_density_data = np.log10(density_data)

def stats(n):
    sample_r = []
    for i in range(0, len(density_data)):
        if np.abs(np.log10(density_data[i]/n)) < 1:
            sample_r.append(reduction_data[i])
    sample_r.sort()
    mean = sum(sample_r)/len(sample_r)
    median = np.quantile(sample_r, .5)
    ten = np.quantile(sample_r, .1)
    return [mean, median, ten]

Npoints = 200
x_n = np.logspace(3, 6, Npoints)
mean_vec = np.zeros(Npoints)
median_vec = np.zeros(Npoints)
ten_vec = np.zeros(Npoints)
for i in range(0, Npoints):
    s = stats(x_n[i])
    mean_vec[i] = s[0]
    median_vec[i] = s[1]
    ten_vec[i] = s[2]

            
rdcut = []
for i in range(0, 1000):
    if density_data[i] > 100:
        rdcut = rdcut + [reduction_data[i]]


fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_subplot(121)
ax1.hist(rdcut, 10)
ax1.set_xlabel('Reduction factor', fontsize = 20)
ax1.set_ylabel('number', fontsize = 20)
plt.setp(ax1.get_xticklabels(), fontsize = 16)
plt.setp(ax1.get_yticklabels(), fontsize = 16)
ax2 = fig.add_subplot(122)
l1, = ax2.plot(x_n, mean_vec)
l2, = ax2.plot(x_n, median_vec)
l3, = ax2.plot(x_n, ten_vec)
plt.legend((l1, l2, l3), ('mean', 'median', '10$^{\\rm th}$ percentile'), loc = "lower right", prop = {'size':14.0}, ncol =1, numpoints = 5, handlelength = 3.5)
plt.xscale('log')
plt.ylim(0.25, 1.05)
ax2.set_ylabel('Reduction factor', fontsize = 20)
ax2.set_xlabel('gas density (hydrogens per cm$^3$)', fontsize = 20)
plt.setp(ax2.get_xticklabels(), fontsize = 16)
plt.setp(ax2.get_yticklabels(), fontsize = 16)
fig.subplots_adjust(left = .1)
fig.subplots_adjust(bottom = .15)
fig.subplots_adjust(top = .98)
fig.subplots_adjust(right = .98)
plt.savefig('histograms/pocket_statistics_ks.pdf')
plt.savefig('histograms/pocket_statistics_ks.png')

