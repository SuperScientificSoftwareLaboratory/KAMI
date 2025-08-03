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

file_path = '../../logs/usage_reg/usage_reg_results.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(
        f"Error: The file at {file_path} was not found. Please check the file path."
    )
    exit(1)

print(df.columns)

groups = df['k'].unique()
compile_1d = df[df['dimension'] == '1d']['reg_count'].values
compile_2d = df[df['dimension'] == '2d']['reg_count'].values
compile_3d = df[df['dimension'] == '3d']['reg_count'].values

theory_1d = 64 * groups / 4 / 32 / 2 + 2 * 32 * groups / 4 / 32 / 2 + 64 * 32 / 4 / 32 / 2 + 14
theory_2d = 2 * 64 * groups / 4 / 32 / 2 + 2 * 32 * groups / 4 / 32 / 2 + 64 * 32 / 4 / 32 / 2 + 14
theory_3d = 2 * 64 * groups / 8 / 32 / 2 + 2 * 32 * groups / 8 / 32 / 2 + 64 * 32 / 4 / 32 / 2 + 15

bar_width = 0.2
index = np.arange(len(groups))

fig, ax = plt.subplots(figsize=(8, 3))

edges = {"bluee": "#2b5dad", "redd": "#a33d3d", "greenn": "#4b7d3a"}
hatch_styles = {
    "redd": ('\\\\', edges["redd"]),
    "greenn": ('xx', edges["greenn"]),
    "bluee": ('//', edges["bluee"])
}

ax.bar(
    index - bar_width,
    theory_1d,
    bar_width,
    label='1D theory',
    alpha=0.5,
    color='lightcoral',
    edgecolor=edges["redd"],
)
ax.bar(
    index - bar_width,
    compile_1d,
    bar_width,
    label='1D compile',
    alpha=0.9,
    color='r',
    hatch=hatch_styles["redd"][0],
    edgecolor=edges["redd"],
)

ax.bar(
    index,
    theory_2d,
    bar_width,
    label='2D theory',
    alpha=0.5,
    color='lightgreen',
    edgecolor=edges["greenn"],
)
ax.bar(
    index,
    compile_2d,
    bar_width,
    label='2D compile',
    alpha=0.9,
    color='g',
    hatch=hatch_styles["greenn"][0],
    edgecolor=edges["greenn"],
)

ax.bar(
    index + bar_width,
    theory_3d,
    bar_width,
    label='3D theory',
    alpha=0.5,
    color='lightblue',
    edgecolor=edges["bluee"],
)
ax.bar(
    index + bar_width,
    compile_3d,
    bar_width,
    label='3D compile',
    alpha=0.9,
    color='b',
    hatch=hatch_styles["bluee"][0],
    edgecolor=edges["bluee"],
)

theory_legend = ax.legend(
    [ax.containers[1], ax.containers[3], ax.containers[5]], ['1D', '2D', '3D'],
    title='Measured',
    bbox_to_anchor=(0.187, 1.04),
    loc='upper left',
    fontsize=18,
    title_fontsize=18,
    labelspacing=0.2,
    borderpad=0.2)

ax.add_artist(theory_legend)

compile_legend = ax.legend(
    [ax.containers[0], ax.containers[2], ax.containers[4]], ['1D', '2D', '3D'],
    title='Theoretical',
    bbox_to_anchor=(-0.012, 1.04),
    loc='upper left',
    fontsize=18,
    title_fontsize=18,
    labelspacing=0.2,
    borderpad=0.2)

ax.set_xticks(index)
ax.set_xticklabels(groups, fontsize=18)

ax.set_xlabel('Dimision of k (m = 64, n = 32)', fontweight='bold', fontsize=18)
ax.set_ylabel('#Registers per thread     ', fontweight='bold', fontsize=18)

ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.axhline(y=255, color='black', linestyle='--', linewidth=2)

ax.text(10.77,
        260,
        'Max reg count',
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=18,
        fontweight='bold')
ax.tick_params(axis='y', labelsize=18)

plt.tight_layout()
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig('usage_reg.pdf', bbox_inches='tight', pad_inches=0.02)

theory_1d = np.minimum(theory_1d, 255)
theory_2d = np.minimum(theory_2d, 255)
theory_3d = np.minimum(theory_3d, 255)

print("Theory values capped at 255:")
print("1D:", theory_1d)
print("2D:", theory_2d)
print("3D:", theory_3d)

percentage_1d = (compile_1d / theory_1d) * 100
percentage_2d = (compile_2d / theory_2d) * 100
percentage_3d = (compile_3d / theory_3d) * 100

results_df = pd.DataFrame({
    'k': groups,
    '1D %': percentage_1d,
    '2D %': percentage_2d,
    '3D %': percentage_3d
})

results_df = results_df.round(2)

print("Percentage of Compile/Theory:")
print(results_df)

avg_1d = np.mean(percentage_1d)
avg_2d = np.mean(percentage_2d)
avg_3d = np.mean(percentage_3d)

print("\nAverage percentages:")
print(f"1D average: {avg_1d:.2f}%")
print(f"2D average: {avg_2d:.2f}%")
print(f"3D average: {avg_3d:.2f}%")
