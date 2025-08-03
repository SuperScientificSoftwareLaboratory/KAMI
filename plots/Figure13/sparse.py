import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Ellipse

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

marker_size = 12
line_width = 3
font_size = 20
legend_font_size = 20

file_paths = {
    "KAMI-1D": "../../logs/sparse/gh200_spgemm_1d_fp16.csv",
    "KAMI-2D": "../../logs/sparse/gh200_spgemm_2d_fp16.csv",
    "KAMI-3D": "../../logs/sparse/gh200_spgemm_3d_fp16.csv"
}

dataframes = {}
for key, path in file_paths.items():
    df = pd.read_csv(path,
                     header=None,
                     names=[
                         'dimension', 'nnzCub', 'mnk', 'n', 'k',
                         'NUM_RANK_BLOCK', 'tflops', 'Block size'
                     ])
    df['mnk'] = df['mnk'].astype(int)
    df['Block size'] = df['Block size'].astype(int)

    df_max = df.loc[df.groupby('mnk')['tflops'].idxmax()]
    df_max = df_max.sort_values(by='mnk')
    dataframes[key] = df_max

plt.figure(figsize=(4, 2))
markers = {'KAMI-1D': 'o', 'KAMI-2D': 's', 'KAMI-3D': '^'}
line_handles = {}

for name, df in dataframes.items():
    line, = plt.plot(df["mnk"],
                     df["tflops"],
                     marker=markers[name],
                     label=name,
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("gh200_spgemm.pdf", bbox_inches='tight', pad_inches=0.01)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Ellipse

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

file_paths = {
    "KAMI-1D": "../../logs/sparse/gh200_spmm_fp16_1d.csv",
    "KAMI-2D": "../../logs/sparse/gh200_spmm_fp16_2d.csv",
    "KAMI-3D": "../../logs/sparse/gh200_spmm_fp16_3d.csv"
}

dataframes = {}
for key, path in file_paths.items():

    if "3D" in key:
        df = pd.read_csv(path,
                         header=None,
                         names=[
                             'dimension', 'mnk', 'rank1', 'NUM_RANK_BLOCK',
                             'time', 'tflops'
                         ])
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=['dimension', 'mnk', 'NUM_RANK_BLOCK', 'time', 'tflops'])

    df['mnk'] = df['mnk'].astype(int)

    df_max = df.loc[df.groupby('mnk')['tflops'].idxmax()]
    df_max = df_max.sort_values(by='mnk')
    dataframes[key] = df_max

plt.figure(figsize=(4, 2))
markers = {'KAMI-1D': 'o', 'KAMI-2D': 's', 'KAMI-3D': '^'}
line_handles = {}

for name, df in dataframes.items():
    line, = plt.plot(df["mnk"],
                     df["tflops"],
                     marker=markers[name],
                     label=name,
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.ylabel("TFLOPS", fontsize=font_size, weight='bold')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')

plt.ylim(51, 280)

ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("gh200_spmm.pdf", bbox_inches='tight', pad_inches=0.01)

fig = plt.figure(figsize=(0, 0))

lines = []
for name in ['KAMI-1D', 'KAMI-2D', 'KAMI-3D']:
    line, = plt.plot([], [],
                     marker=markers[name],
                     label=name,
                     markersize=marker_size,
                     linewidth=line_width)
    lines.append(line)

legend = plt.legend(
    lines,
    ['KAMI-1D', 'KAMI-2D', 'KAMI-3D'],
    fontsize=legend_font_size,
    ncol=3,
    frameon=False,
    labelspacing=0.1,
    borderpad=0.1,
)

plt.axis('off')

plt.savefig("sparse_legend.pdf", bbox_inches='tight', pad_inches=0.01)
