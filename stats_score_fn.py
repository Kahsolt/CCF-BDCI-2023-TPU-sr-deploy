#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/20 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from run_utils import get_score, OUT_PATH


x = np.linspace(4.0, 5.5, 100)
y = np.linspace(0.4, 1.2, 100)
z = np.empty([len(x), len(y)], dtype=np.float32)

for i in range(z.shape[0]):
  for j in range(z.shape[1]):
    z[i, j] = get_score(x[i], y[j])

plt.clf()
sns.heatmap(z, cbar=True)
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('time')
plt.ylabel('niqe')
plt.tight_layout()
fp = OUT_PATH / 'score_fn.png'
print(f'>> save to {fp}')
plt.savefig(fp, dpi=600)
