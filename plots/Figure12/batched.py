import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
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

kami_path = "../../logs/batched_gemm/H200/KAMI_batch_fp64.csv"
magma_path = "../../logs/batched_gemm/H200/MAGMA_batch_fp64.csv"

mnk_keep = [16, 32, 48, 64, 96, 128]
if not os.path.exists(kami_path):
    raise FileNotFoundError(f"KAMI file not found: {kami_path}")

kami_all = pd.read_csv(kami_path, header=None)
kami_all.columns = [
    'dimension', 'm', 'n', 'k', 'batchcount', 'tflops', 'blocksize'
]
kami_all = kami_all[kami_all['batchcount'].isin([1000, 10000])]
kami_all['mnk'] = kami_all['m']
kami_best = kami_all.sort_values('tflops', ascending=False).drop_duplicates(
    ['m', 'n', 'k', 'batchcount'])
has_magma_data = False
if os.path.exists(magma_path):
    try:
        magma_df = pd.read_csv(magma_path, header=None)
        if not magma_df.empty:
            has_magma_data = True
            magma_df.columns = [
                'batchcount', 'm', 'n', 'k',
                'magma_gflops', 'magma_time',
                'cublas_gflops', 'cublas_time'
            ]
            magma_df = magma_df[magma_df['batchcount'].isin([1000, 10000])]
            magma_df['mnk'] = magma_df['m']
            magma_df['cublas_tflops'] = magma_df['cublas_gflops'] / 1000
            magma_df['magma_tflops'] = magma_df['magma_gflops'] / 1000
    except pd.errors.EmptyDataError:
        print(f"Warning: {magma_path} is empty. Skipping MAGMA/cuBLAS plots.")
    except pd.errors.ParserError:
        print(f"Warning: {magma_path} contains invalid CSV data. Skipping MAGMA/cuBLAS plots.")
else:
    print(f"Warning: {magma_path} does not exist. Only plotting KAMI.")


def get_sorted(df, batch):
    return df[(df['batchcount'] == batch)
              & (df['mnk'].isin(mnk_keep))].sort_values('mnk')


kami_1000 = get_sorted(kami_best, 1000)
kami_10000 = get_sorted(kami_best, 10000)
if has_magma_data:
    cublas_1000 = get_sorted(magma_df, 1000)
    cublas_10000 = get_sorted(magma_df, 10000)
    magma_1000 = cublas_1000[['mnk', 'magma_tflops']]
    magma_10000 = cublas_10000[['mnk', 'magma_tflops']]

    speedup_1000_magma = kami_1000['tflops'].values / magma_1000[
        'magma_tflops'].values
    speedup_1000_cublas = kami_1000['tflops'].values / cublas_1000[
        'cublas_tflops'].values
    speedup_10000_magma = kami_10000['tflops'].values / magma_10000[
        'magma_tflops'].values
    speedup_10000_cublas = kami_10000['tflops'].values / cublas_10000[
        'cublas_tflops'].values

print("\n===== KAMI Data (Batch = 1000) =====")
print(kami_1000)
print("\n===== KAMI Data (Batch = 10000) =====")
print(kami_10000)
if has_magma_data:
    print("\n===== Speedup (Batch = 1000) =====")
    print("Matrix Size:", kami_1000['mnk'].values)
    print("KAMI / MAGMA:", np.round(speedup_1000_magma, 2))
    print("KAMI / cuBLAS:", np.round(speedup_1000_cublas, 2))
    print("\n===== Speedup (Batch = 10000) =====")
    print("Matrix Size:", kami_10000['mnk'].values)
    print("KAMI / MAGMA:", np.round(speedup_10000_magma, 2))
    print("KAMI / cuBLAS:", np.round(speedup_10000_cublas, 2))

    print("\n===== Average Speedup =====")
    print("Batch = 1000:")
    print(
        f"vs MAGMA: {np.mean(speedup_1000_magma):.2f}x (max: {np.max(speedup_1000_magma):.2f}x)"
    )
    print(
        f"vs cuBLAS: {np.mean(speedup_1000_cublas):.2f}x (max: {np.max(speedup_1000_cublas):.2f}x)"
    )
    print("\nBatch = 10000:")
    print(
        f"vs MAGMA: {np.mean(speedup_10000_magma):.2f}x (max: {np.max(speedup_10000_magma):.2f}x)"
    )
    print(
        f"vs cuBLAS: {np.mean(speedup_10000_cublas):.2f}x (max: {np.max(speedup_10000_cublas):.2f}x)"
    )

fig, ax = plt.subplots(figsize=(9, 2.7))

ax.fill_between(kami_1000['mnk'],
                kami_1000['tflops'],
                kami_10000['tflops'],
                color='C1',
                alpha=0.2)
if has_magma_data:
    ax.fill_between(cublas_1000['mnk'],
                    cublas_1000['cublas_tflops'],
                    cublas_10000['cublas_tflops'],
                    color='C0',
                    alpha=0.2)
    ax.fill_between(magma_1000['mnk'],
                    magma_1000['magma_tflops'],
                    magma_10000['magma_tflops'],
                    color='C2',
                    alpha=0.2)

    ax.plot(cublas_1000['mnk'],
            cublas_1000['cublas_tflops'],
            color='C0',
            linestyle='-',
            linewidth=line_width,
            marker='o',
            markersize=marker_size)
    ax.plot(cublas_10000['mnk'],
            cublas_10000['cublas_tflops'],
            color='C0',
            linestyle='--',
            linewidth=line_width,
            marker='o',
            markersize=marker_size)
    ax.plot(magma_1000['mnk'],
        magma_1000['magma_tflops'],
        color='C2',
        linestyle='-',
        linewidth=line_width,
        marker='^',
        markersize=marker_size)
    ax.plot(magma_10000['mnk'],
            magma_10000['magma_tflops'],
            color='C2',
            linestyle='--',
            linewidth=line_width,
            marker='^',
            markersize=marker_size)

ax.plot(kami_1000['mnk'],
        kami_1000['tflops'],
        color='C1',
        linestyle='-',
        linewidth=line_width,
        marker='s',
        markersize=marker_size)
ax.plot(kami_10000['mnk'],
        kami_10000['tflops'],
        color='C1',
        linestyle='--',
        linewidth=line_width,
        marker='s',
        markersize=marker_size)


ax.set_xlabel('Matrix order', fontsize=font_size, weight='bold')
ax.set_ylabel('TFLOPS', fontsize=font_size, weight='bold')
ax.tick_params(labelsize=font_size)

if has_magma_data:
    handles_batch1000 = [
        Line2D([0], [0],
            color='C2',
            linestyle='-',
            marker='^',
            label='MAGMA',
            linewidth=line_width,
            markersize=marker_size),
        Line2D([0], [0],
            color='C0',
            linestyle='-',
            marker='o',
            label='cuBLAS',
            linewidth=line_width,
            markersize=marker_size),
        Line2D([0], [0],
            color='C1',
            linestyle='-',
            marker='s',
            label='KAMI',
            linewidth=line_width,
            markersize=marker_size),
    ]

    handles_batch10000 = [
        Line2D([0], [0],
            color='C2',
            linestyle='--',
            marker='^',
            label='MAGMA',
            linewidth=line_width,
            markersize=marker_size),
        Line2D([0], [0],
            color='C0',
            linestyle='--',
            marker='o',
            label='cuBLAS',
            linewidth=line_width,
            markersize=marker_size),
        Line2D([0], [0],
            color='C1',
            linestyle='--',
            marker='s',
            label='KAMI',
            linewidth=line_width,
            markersize=marker_size),
    ]

    legend1 = fig.legend(
        handles=[Line2D([0], [0], color='none', label='#Batches=1000  ')] +
        handles_batch1000,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.47),
        ncol=4,
        fontsize=legend_font_size,
        frameon=False,
        handletextpad=0.5,
        handlelength=2.4,
        columnspacing=0.6)

    legend2 = fig.legend(
        handles=[Line2D([0], [0], color='none', label='#Batches=10000')] +
        handles_batch10000,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.37),
        ncol=4,
        fontsize=legend_font_size,
        frameon=False,
        handletextpad=0.5,
        handlelength=2.4,
        columnspacing=0.6)
else:
    handles_batch1000 = [
        Line2D([0], [0],
            color='C1',
            linestyle='-',
            marker='s',
            label='KAMI',
            linewidth=line_width,
            markersize=marker_size),
    ]

    handles_batch10000 = [
        Line2D([0], [0],
            color='C1',
            linestyle='--',
            marker='s',
            label='KAMI',
            linewidth=line_width,
            markersize=marker_size),
    ]

    legend1 = fig.legend(
        handles=[Line2D([0], [0], color='none', label='#Batches=1000  ')] + handles_batch1000,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.47),
        ncol=2, 
        fontsize=legend_font_size,
        frameon=False,
        handletextpad=0.5,
        handlelength=2.4,
        columnspacing=0.6)

    legend2 = fig.legend(
        handles=[Line2D([0], [0], color='none', label='#Batches=10000')] + handles_batch10000,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.37),
        ncol=2,
        fontsize=legend_font_size,
        frameon=False,
        handletextpad=0.5,
        handlelength=2.4,
        columnspacing=0.6)


plt.tight_layout(rect=[0, 0, 1, 1.25])
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("GH200_FP64_batch_comparison.pdf",
            bbox_inches="tight",
            pad_inches=0.02)
