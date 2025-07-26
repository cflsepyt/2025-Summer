import numpy as np
import matplotlib.pyplot as plt

# cases
case_list = ['LW_200_sst1K', 'LW_200_sst2.5K', 'LW_300_sst1K', 'LW_300_sst2.5K']

# load file
growth_rate, freq = np.loadtxt('test.txt', usecols=(1, 2), delimiter=',', unpack=True)


# plot
fig, ax = plt.subplots(figsize=(6, 5))

# scatter plot
ax.scatter(growth_rate, freq, color='blue', s=50)

# 根據索引分散標籤方向
offsets = [(-40, -15), (-10, 10), (20, -15), (-40, -15)]  # 右上、左上、右下、左下

for i, case in enumerate(case_list):
    dx, dy = offsets[i % len(offsets)]  # 依序選不同偏移
    ax.annotate(
        case,
        xy=(growth_rate[i], freq[i]),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        ha='center'
    )

# limit
ax.set_xlim(-0.0006, 0.0006)
ax.set_ylim(0, 0.06)

# labels
ax.set_xlabel('Growth rate')
ax.set_ylabel('Frequency')
# ax.set_title('Growth rate vs Frequency')

plt.tight_layout()
plt.grid(True, linestyle='--')
plt.savefig('growth_freq.png', dpi=300, bbox_inches='tight')