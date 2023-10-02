import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

normal = np.random.normal(0, 1, 10000) # loc, scale, size
quartiles = pd.DataFrame(normal).quantile([0.25, 0.5, 0.75, 1])[0]
fig, axs = plt.subplots(nrows=2)
fig.set_size_inches(14, 8)
# Boxplot of Normal distribution
plot1 = sns.boxplot(normal, ax=axs[0])
plot1.set(xlim=(-4, 4))
# Normal distribution
plot2 = sns.histplot(normal, ax=axs[1])
plot2.set(xlim=(-4, 4))
# Median line
plt.axvline(np.median(normal), color='r', linestyle='dashed', linewidth=2)
for i, q in enumerate(quartiles):
    # Quartile i line
    plt.axvline(q, color='g', linestyle='dotted', linewidth=2)

plt.figure(figsize=(12,6))
sns.boxplot(normal)

normal[(normal >= -3) & (normal <= 3)]
np.array([-0.13228601, -0.43618127,  0.49768295, ..., -0.92085958,
        1.1504461 , -0.15994229])
plt.figure(figsize=(12,6))
sns.boxplot(normal[(normal >= -3) & (normal <= 3)])

q1 = pd.DataFrame(normal).quantile(0.25)[0]
q3 = pd.DataFrame(normal).quantile(0.75)[0]
iqr = q3 - q1 #Interquartile range
fence_low = q1 - (1.5*iqr)
fence_high = q3 + (1.5*iqr)

# "Outside" boxplot Reviews
normal[(normal < fence_low) | (normal > fence_high)].shape[0]

plt.figure(figsize=(12,6))
sns.boxplot(normal[(normal >= fence_low) & (normal <= fence_high)])