import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

mnk_keep = [16, 32, 64, 128]
mnk_speedup = [16, 32, 64]
marker_size = 12
line_width = 3
font_size = 24
legend_font_size = 20

file_paths = {
    "KAMI-1D":
    "../../logs/block_gemm/square/H200/fp64_block_square_1d_H200.csv",
    "KAMI-2D":
    "../../logs/block_gemm/square/H200/fp64_block_square_2d_H200.csv",
    "KAMI-3D":
    "../../logs/block_gemm/square/H200/fp64_block_square_3d_H200.csv",
    "cuBLASDx": "../../logs/block_gemm/square/H200/fp64_cublasdx.csv",
    "CUTLASS": "../../logs/block_gemm/square/H200/fp64_CUTLASS.csv"
}


def load_and_filter(path, name):
    if name == "cuBLASDx":
        df = pd.read_csv(
            path,
            header=None,
            names=["mnk", "n", "k", "Block size", "gflops", "tflops"])
        df['mnk'] = df['mnk'].astype(int)
    elif name == "CUTLASS":
        df = pd.read_csv(
            path,
            header=None,
            names=["m", "n", "k", "blocksize", "gridsize", "gflops", "tflops"])
        df = (df[(df["m"] == df["n"]) & (df["m"] == df["k"])])
        df = df[df["m"].isin(mnk_keep)]
        df["mnk"] = df["m"]
        best = df.groupby("mnk")["tflops"].median().reset_index()
        return best.sort_values("mnk")
    else:
        df = pd.read_csv(path,
                         header=None,
                         names=[
                             "dimension", "mnk", "NUM_RANK1", "NUM_RANK2",
                             "tflops", "Block size"
                         ])
        df['mnk'] = df['mnk'].astype(int)

    df = df[df['mnk'].isin(mnk_keep)]
    best = df.groupby("mnk")['tflops'].max().reset_index()
    return best.sort_values("mnk")


filtered_data = {
    name: load_and_filter(path, name)
    for name, path in file_paths.items()
}

plt.figure(figsize=(4, 5))
markers = {
    'KAMI-1D': 'o',
    'KAMI-2D': 's',
    'KAMI-3D': '^',
    'cuBLASDx': 'D',
    'CUTLASS': 'X'
}
line_handles = {}

for name, df in filtered_data.items():
    df = df[df['mnk'].isin(mnk_keep)]
    line, = plt.plot(df['mnk'],
                     df['tflops'],
                     label=name,
                     marker=markers[name],
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

print("\n====== Speedup Analysis (KAMI vs. CUTLASS) ======")
cutlass_df = filtered_data["CUTLASS"]

for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    kami_df = filtered_data[name]
    common_mnk = kami_df['mnk'].isin(cutlass_df['mnk'])
    kami_df = kami_df[common_mnk]
    matched_cutlass = cutlass_df[cutlass_df['mnk'].isin(kami_df['mnk'])]

    kami_df = kami_df.sort_values("mnk")
    matched_cutlass = matched_cutlass.sort_values("mnk")

    speedup = kami_df['tflops'].values / matched_cutlass['tflops'].values

    print(f"\n{name} speedup over CUTLASS:")
    for i, mnk in enumerate(kami_df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

print("\n====== Speedup Analysis (KAMI vs. cuBLASDx) ======")
dx_df = filtered_data["cuBLASDx"]

for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    kami_df = filtered_data[name]
    common_mnk = kami_df['mnk'].isin(dx_df['mnk'])
    kami_df = kami_df[common_mnk]
    matched_dx = dx_df[dx_df['mnk'].isin(kami_df['mnk'])]

    kami_df = kami_df.sort_values("mnk")
    matched_dx = matched_dx.sort_values("mnk")

    speedup = kami_df['tflops'].values / matched_dx['tflops'].values

    print(f"\n{name} speedup over cuBLASDx:")
    for i, mnk in enumerate(kami_df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.ylabel("TFLOPS", fontsize=font_size, weight='bold')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:3.0f}'))
plt.grid(True, linestyle='--')

ax = plt.gca()
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_color('black')

plt.savefig("GH200_FP64.pdf", bbox_inches='tight', pad_inches=0.02)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

color_pattern = sns.color_palette()

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

marker_size = 12
line_width = 3
font_size = 24
legend_font_size = 20
mnk_keep = [16, 32, 64, 128, 192]

file_paths = {
    "KAMI-1D":
    "../../logs/block_gemm/square/H200/fp16_block_square_1d_H200.csv",
    "KAMI-2D":
    "../../logs/block_gemm/square/H200/fp16_block_square_2d_H200.csv",
    "KAMI-3D":
    "../../logs/block_gemm/square/H200/fp16_block_square_3d_H200.csv",
    "cuBLASDx": "../../logs/block_gemm/square/H200/fp16_cublasdx.csv",
    "CUTLASS": "../../logs/block_gemm/square/H200/fp16_CUTLASS.csv"
}


def load_and_filter(path, name):
    if name == "cuBLASDx":
        df = pd.read_csv(
            path,
            header=None,
            names=["mnk", "n", "k", "Block size", "gflops", "tflops"])
        df['mnk'] = df['mnk'].astype(int)
    elif name == "CUTLASS":
        df = pd.read_csv(
            path,
            header=None,
            names=["m", "n", "k", "blocksize", "gridsize", "gflops", "tflops"])
        df = df[(df["m"] == df["n"]) & (df["m"] == df["k"])]
        df = df[df["m"].isin(mnk_keep)]
        df["mnk"] = df["m"]
        best = df.groupby("mnk")["tflops"].median().reset_index()
        return best.sort_values("mnk")
    else:
        df = pd.read_csv(path,
                         header=None,
                         names=[
                             "dimension", "mnk", "NUM_RANK1", "NUM_RANK2",
                             "tflops", "Block size"
                         ])
        df['mnk'] = df['mnk'].astype(int)

    df = df[df['mnk'].isin(mnk_keep)]
    best = df.groupby("mnk")['tflops'].max().reset_index()
    return best.sort_values("mnk")


filtered_data = {
    name: load_and_filter(path, name)
    for name, path in file_paths.items()
}

plt.figure(figsize=(4, 5))
markers = {
    'KAMI-1D': 'o',
    'KAMI-2D': 's',
    'KAMI-3D': '^',
    'cuBLASDx': 'D',
    'CUTLASS': 'X'
}
line_handles = {}

for name, df in filtered_data.items():
    line, = plt.plot(df['mnk'],
                     df['tflops'],
                     label=name,
                     marker=markers[name],
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

cutlass_ref = filtered_data["CUTLASS"]
cutlass_vals = cutlass_ref[cutlass_ref["mnk"].isin(mnk_keep)]["tflops"].values

print("\n====== Speedup Analysis (KAMI vs. CUTLASS) ======")
cutlass_df = filtered_data["CUTLASS"]

for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    kami_df = filtered_data[name]
    common_mnk = kami_df['mnk'].isin(cutlass_df['mnk'])
    kami_df = kami_df[common_mnk]
    matched_cutlass = cutlass_df[cutlass_df['mnk'].isin(kami_df['mnk'])]

    kami_df = kami_df.sort_values("mnk")
    matched_cutlass = matched_cutlass.sort_values("mnk")

    speedup = kami_df['tflops'].values / matched_cutlass['tflops'].values

    print(f"\n{name} speedup over CUTLASS:")
    for i, mnk in enumerate(kami_df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

print("\n====== Speedup Analysis (KAMI vs. cuBLASDx) ======")
dx_df = filtered_data["cuBLASDx"]

for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    kami_df = filtered_data[name]
    common_mnk = kami_df['mnk'].isin(dx_df['mnk'])
    kami_df = kami_df[common_mnk]
    matched_dx = dx_df[dx_df['mnk'].isin(kami_df['mnk'])]

    kami_df = kami_df.sort_values("mnk")
    matched_dx = matched_dx.sort_values("mnk")

    speedup = kami_df['tflops'].values / matched_dx['tflops'].values

    print(f"\n{name} speedup over cuBLASDx:")
    for i, mnk in enumerate(kami_df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("GH200_FP16.pdf", bbox_inches='tight', pad_inches=0.02)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

mnk_keep = [16, 32, 64, 128, 192]
mnk_speedup = [16, 32, 64]
marker_size = 12
line_width = 3
font_size = 24
legend_font_size = 20

file_paths = {
    "KAMI-1D":
    "../../logs/block_gemm/square/5090/fp16_block_square_1d_5090.csv",
    "KAMI-2D":
    "../../logs/block_gemm/square/5090/fp16_block_square_2d_5090.csv",
    "KAMI-3D":
    "../../logs/block_gemm/square/5090/fp16_block_square_3d_5090.csv",
    "cuBLASDx": "../../logs/block_gemm/square/5090/cublasdx.csv",
    "CUTLASS": "../../logs/block_gemm/square/5090/CUTLASS.csv"
}


def load_and_filter(path, name):
    if name == "cuBLASDx":
        df = pd.read_csv(
            path,
            header=None,
            names=["mnk", "n", "k", "Block size", "gflops", "tflops"])
        df['mnk'] = df['mnk'].astype(int)
    elif name == "CUTLASS":
        df = pd.read_csv(
            path,
            header=None,
            names=["m", "n", "k", "blocksize", "gridsize", "gflops", "tflops"])
        df = df[(df["m"] == df["n"]) & (df["m"] == df["k"])]
        df = df[df["m"].isin(mnk_keep)]
        df["mnk"] = df["m"]
        best = df.groupby("mnk")["tflops"].median().reset_index()
        return best.sort_values("mnk")
    else:
        df = pd.read_csv(path,
                         header=None,
                         names=[
                             "dimension", "mnk", "n", "k", "NUM_RANK1",
                             "NUM_RANK2", "tflops", "Block size"
                         ])
        df['mnk'] = df['mnk'].astype(int)

    df = df[df['mnk'].isin(mnk_keep)]
    best = df.loc[df.groupby("mnk")['tflops'].idxmax()]
    return best.sort_values("mnk")


filtered_data = {
    name: load_and_filter(path, name)
    for name, path in file_paths.items()
}

plt.figure(figsize=(4, 5))
markers = {
    'KAMI-1D': 'o',
    'KAMI-2D': 's',
    'KAMI-3D': '^',
    'cuBLASDx': 'D',
    'CUTLASS': 'X'
}
line_handles = {}

for name, df in filtered_data.items():
    df = df[df['mnk'].isin(mnk_keep)]
    line, = plt.plot(df['mnk'],
                     df['tflops'],
                     label=name,
                     marker=markers[name],
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

ref_vals_cutlass = filtered_data["CUTLASS"][
    filtered_data["CUTLASS"]['mnk'].isin(mnk_keep)]['tflops'].values

print("\n====== Speedup Analysis (KAMI vs. CUTLASS) ======")
for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    df = filtered_data[name][filtered_data[name]['mnk'].isin(mnk_keep)]
    speedup = df['tflops'].values / ref_vals_cutlass

    print(f"\n{name} speedup over CUTLASS:")
    for i, mnk in enumerate(df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

ref_vals_dx = filtered_data["cuBLASDx"][filtered_data["cuBLASDx"]['mnk'].isin(
    mnk_speedup)]['tflops'].values

print("\n====== Speedup Analysis (KAMI vs. cuBLASDx) ======")
for name in ["KAMI-1D", "KAMI-2D", "KAMI-3D"]:
    df = filtered_data[name][filtered_data[name]['mnk'].isin(mnk_speedup)]
    speedup = df['tflops'].values / ref_vals_dx

    print(f"\n{name} speedup over cuBLASDx:")
    for i, mnk in enumerate(df['mnk']):
        print(f"Matrix size {mnk}: {speedup[i]:.2f}x")
    print(f"{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')
ax = plt.gca()
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_color('black')

plt.savefig("5090_FP16.pdf", bbox_inches='tight', pad_inches=0.02)

from matplotlib.ticker import FuncFormatter
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
font_size = 24
legend_font_size = 20

file_paths = {
    "KAMI-1D": "../../logs/block_gemm/square/amd/fp16_block_square_1d_amd.csv",
    "KAMI-2D": "../../logs/block_gemm/square/amd/fp16_block_square_2d_amd.csv",
    "KAMI-3D": "../../logs/block_gemm/square/amd/fp16_block_square_3d_amd.csv"
}

dataframes = {}
for key, path in file_paths.items():
    df = pd.read_csv(path,
                     header=None,
                     names=[
                         "dimension", "mnk", "n", "k", "NUM_RANK1",
                         "NUM_RANK2", "tflops", "Block size"
                     ])
    df['mnk'] = df['mnk'].astype(int)
    df['Block size'] = df['Block size'].astype(int)

    df_max = df.loc[df.groupby('mnk')['tflops'].idxmax()]
    df_max = df_max.sort_values(by='mnk')
    dataframes[key] = df_max

plt.figure(figsize=(4, 5))
markers = {'KAMI-1D': 'o', 'KAMI-2D': 's', 'KAMI-3D': '^'}
line_handles = {}

mnk_keep = [16, 32, 64, 128, 192]

for name, df in dataframes.items():
    df = df[df['mnk'].isin(mnk_keep)]
    line, = plt.plot(df["mnk"],
                     df["tflops"],
                     marker=markers[name],
                     label=name,
                     markersize=marker_size,
                     linewidth=line_width)
    line_handles[name] = line

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:3.0f}'))

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("amd_FP16.pdf", bbox_inches='tight', pad_inches=0.02)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })
marker_size = 12
line_width = 3
font_size = 24
legend_font_size = 20
df_1d = pd.read_csv(
    '../../logs/block_gemm/square/intel/fp16_block_square_1d_intel.csv',
    header=None)
df_2d = pd.read_csv(
    '../../logs/block_gemm/square/intel/fp16_block_square_2d_intel.csv',
    header=None)
df_3d = pd.read_csv(
    '../../logs/block_gemm/square/intel/fp16_block_square_3d_intel.csv',
    header=None)
df_basic = pd.read_csv(
    '../../logs/block_gemm/square/intel/fp16_basic_intel.csv', header=None)

df_1d.columns = df_2d.columns = df_3d.columns = [
    'dimension', 'm', 'n', 'k', 'rank', 'time', 'tflops', 'blocksize'
]
df_basic.columns = ['m', 'gflops']
df_basic['tflops'] = df_basic['gflops'] / 1000

mnk_keep = [16, 32, 64, 128, 192]
df_1d = df_1d[df_1d['m'].isin(mnk_keep)]
df_2d = df_2d[df_2d['m'].isin(mnk_keep)]
df_3d = df_3d[df_3d['m'].isin(mnk_keep)]
df_basic = df_basic[df_basic['m'].isin(mnk_keep)]


def get_best(df):
    return df.groupby(['m', 'n', 'k'],
                      as_index=False)['tflops'].max()[['m', 'tflops'
                                                       ]].sort_values('m')


df_1d_sorted = get_best(df_1d)
df_2d_sorted = get_best(df_2d)
df_3d_sorted = get_best(df_3d)
df_basic_sorted = df_basic.sort_values('m')[['m', 'tflops']]

for df, name in zip([df_1d_sorted, df_2d_sorted, df_3d_sorted],
                    ['KAMI-1D', 'KAMI-2D', 'KAMI-3D']):
    speedup = df['tflops'].values / df_basic_sorted['tflops'].values
    print(f"\n{name} average speedup: {speedup.mean():.2f}x")
    print(f"{name} maximum speedup: {speedup.max():.2f}x")

plt.figure(figsize=(4, 5))

markers = {'KAMI-1D': 'o', 'KAMI-2D': 's', 'KAMI-3D': '^', 'SYCL-Bench': 'P'}
line_handles = {}

line_handles['KAMI-1D'], = plt.plot(df_1d_sorted['m'],
                                    df_1d_sorted['tflops'],
                                    marker=markers['KAMI-1D'],
                                    label='KAMI-1D',
                                    markersize=marker_size,
                                    linewidth=line_width)
line_handles['KAMI-2D'], = plt.plot(df_2d_sorted['m'],
                                    df_2d_sorted['tflops'],
                                    marker=markers['KAMI-2D'],
                                    label='KAMI-2D',
                                    markersize=marker_size,
                                    linewidth=line_width)
line_handles['KAMI-3D'], = plt.plot(df_3d_sorted['m'],
                                    df_3d_sorted['tflops'],
                                    marker=markers['KAMI-3D'],
                                    label='KAMI-3D',
                                    markersize=marker_size,
                                    linewidth=line_width)
line_handles['SYCL-Bench'], = plt.plot(df_basic_sorted['m'],
                                       df_basic_sorted['tflops'],
                                       marker=markers['SYCL-Bench'],
                                       label='SYCL-Bench',
                                       markersize=marker_size,
                                       linewidth=line_width,
                                       color='slategray')

plt.xlabel("Matrix order", fontsize=font_size, weight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:3.1f}'))
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig('intel_FP16.pdf', bbox_inches='tight', pad_inches=0.02)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

marker_size = 12
line_width = 3
legend_font_size = 20

markers = {
    'KAMI-1D': 'o',
    'KAMI-2D': 's',
    'KAMI-3D': '^',
    'cuBLASDx': 'D',
    'CUTLASS': 'X',
    'SYCL-Bench': 'P'
}

colors = {
    'cuBLASDx': color_pattern[3],
    'CUTLASS': color_pattern[4],
    'SYCL-Bench': 'slategrey',
    'KAMI-1D': color_pattern[0],
    'KAMI-2D': color_pattern[1],
    'KAMI-3D': color_pattern[2]
}

fig = plt.figure(figsize=(8, 0.5))

lines = []
labels = ['cuBLASDx', 'CUTLASS', 'SYCL-Bench', 'KAMI-1D', 'KAMI-2D', 'KAMI-3D']

for name in labels:
    line, = plt.plot([], [],
                     marker=markers[name],
                     label=name,
                     color=colors[name],
                     markersize=marker_size,
                     linewidth=line_width)
    lines.append(line)

legend = plt.legend(lines,
                    labels,
                    fontsize=legend_font_size,
                    ncol=6,
                    frameon=False,
                    labelspacing=0.1,
                    borderpad=0.1,
                    columnspacing=1.0,
                    handletextpad=0.4)

plt.axis('off')

plt.savefig("gemm_legend.pdf", bbox_inches='tight', pad_inches=0.02)
