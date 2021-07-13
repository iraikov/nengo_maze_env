import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a = 4
r = np.clip(skewnorm.rvs(a, loc=1, scale=1, size=1000), 0., None)
print(np.min(r), np.max(r))

ax.hist(r, density=True, histtype='stepfilled', alpha=0.5)
plt.show()

