import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Ellipse

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold',
                  'mathtext.fontset': 'stix'
              })

file_path1 = '../../logs/usage_cycle/usage_cycle_5090.csv'
file_path2 = '../../logs/usage_cycle/usage_cycle_GH200.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

print(df1.columns)
print(df2.columns)

groups = [
    "Tcomm\n(1D)", "Tcomm\n(2D)", "Tcomm\n(3D)", "Tcomp\n(1D)", "Tcomp\n(2D)",
    "Tcomp\n(3D)"
]


def find_closest_value(series, target):
    return series.iloc[(series - target).abs().argmin()]


theory_5090 = [2304, 1536, 1536, 4096, 4096, 4096]

values_5090 = [
    find_closest_value(df1[df1['dimension'] == '1d']['communication_time'],
                       theory_5090[0]),
    find_closest_value(df1[df1['dimension'] == '2d']['communication_time'],
                       theory_5090[1]),
    find_closest_value(df1[df1['dimension'] == '3d']['communication_time'],
                       theory_5090[2]),
    find_closest_value(df1[df1['dimension'] == '1d']['computation_time'],
                       4096),
    find_closest_value(df1[df1['dimension'] == '2d']['computation_time'],
                       4096),
    find_closest_value(df1[df1['dimension'] == '3d']['computation_time'], 4096)
]

theory_GH200 = [2304, 1536, 1536, 1024, 1024, 1024]

values_GH200 = [
    find_closest_value(df2[df2['dimension'] == '1d']['communication_time'],
                       theory_GH200[0]),
    find_closest_value(df2[df2['dimension'] == '2d']['communication_time'],
                       theory_GH200[1]),
    find_closest_value(df2[df2['dimension'] == '3d']['communication_time'],
                       theory_GH200[2]),
    find_closest_value(df2[df2['dimension'] == '1d']['computation_time'],
                       1024),
    find_closest_value(df2[df2['dimension'] == '2d']['computation_time'],
                       1024),
    find_closest_value(df2[df2['dimension'] == '3d']['computation_time'], 1024)
]

bar_width = 0.2
index = np.arange(len(groups))

edges = {"bluee": "#2b5dad", "redd": "#a33d3d", "greenn": "#4b7d3a"}
hatch_styles = {
    "redd": ('\\\\', edges["redd"]),
    "greenn": ('xx', edges["greenn"]),
    "bluee": ('//', edges["bluee"])
}

fig, ax = plt.subplots(figsize=(8, 3))

h200_bars = ax.bar(
    index - bar_width / 2,
    values_GH200,
    bar_width,
    label='Measured on GH200',
    alpha=0.7,
    color='#f4a7a7',
    hatch=hatch_styles["redd"][0],
    edgecolor=edges['redd'],
    linewidth=2,
)

rtx5090_bars = ax.bar(
    index + bar_width / 2,
    values_5090,
    bar_width,
    label='Measured on 5090',
    alpha=0.7,
    color='#9fcff3',
    hatch=hatch_styles["bluee"][0],
    edgecolor=edges['bluee'],
    linewidth=1.5,
)

ax.scatter(index + bar_width / 2,
           theory_5090,
           color='darkblue',
           marker='^',
           s=100,
           label='Theoretical on 5090')
ax.scatter(index - bar_width / 2,
           theory_GH200,
           color='darkred',
           marker='*',
           s=100,
           label='Theoretical on GH200')

blank_line = plt.Line2D([], [], color='none', label='     ')

handles1, labels1 = ax.get_legend_handles_labels()
measured_legend = ax.legend([blank_line, h200_bars, rtx5090_bars],
                            ['         ', 'GH200', '5090'],
                            ncol=3,
                            loc='upper left',
                            bbox_to_anchor=(-0.0096, 0.836),
                            columnspacing=0.2,
                            handletextpad=0.3,
                            fontsize=18,
                            title_fontsize=18,
                            borderpad=0.3)

ax.text(0.015,
        0.66,
        'Measured:',
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        fontfamily='Times New Roman',
        zorder=10)
ax.text(0.015,
        0.86,
        'Theoretical:',
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        fontfamily='Times New Roman',
        zorder=10)

ax.add_artist(measured_legend)

theory_legend = ax.legend([blank_line, handles1[1], handles1[0]],
                          ['         ', 'GH200', '5090'],
                          ncol=3,
                          loc='upper left',
                          bbox_to_anchor=(-0.0096, 1.03),
                          columnspacing=0.2,
                          handletextpad=0.3,
                          fontsize=18,
                          title_fontsize=18,
                          borderpad=0.3)

groups = [
    r'$T_{cm}$(1D)', r'$T_{cm}$(2D)', r'$T_{cm}$(3D)', r'$T_{cp}$(1D)',
    r'$T_{cp}$(2D)', r'$T_{cp}$(3D)'
]

ax.set_xticks(index)
ax.set_xticklabels(groups, fontsize=18)
ax.set_ylabel('#Cycles', fontweight='bold', fontsize=18)

ax.tick_params(axis='y', labelsize=18)

ax.set_ylim(0, 5600)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig('usage_cycle.pdf', bbox_inches='tight', pad_inches=0.02)
