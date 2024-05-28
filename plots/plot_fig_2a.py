#%% read csv
import pandas as pd
# df = pd.read_csv('./data/model_predictions_after_corr_stripped/fig_2c_combined.csv')
df = pd.read_csv('/home/skowshik/publication_ADRD_repo/adrd_tool/model_predictions_stripped_MNI_swinunetr/fig_2c_combined.csv')

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import rc, rcParams

rc('axes', linewidth=0.5)
rc('font', size=7)
rcParams['font.family'] = 'Arial'
linewidth =  0.5

feature_displays = {
    'bat': 'NP',
    'updrs': 'UPDRS',
    'npiq': 'NPI-Q',
    'gds': 'GDS',
    'faq': 'FAQ',
    'img_MRI': 'MRI',
} 

labels = ['NC', 'MCI', 'DE']
palette_label = dict(zip(
    labels,
    [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
))
features = ['bat', 'faq', 'npiq', 'gds', 'updrs', 'img_MRI'][::-1]
# import random
# random.shuffle(features)
n_features = len(features)
n_labels = len(labels)

# compute auroc
def eval_auroc(label):
    def helper(grp):
        return roc_auc_score(
            grp['{}_label'.format(label)], 
            grp['{}_prob'.format(label)]
        )
    return df.groupby(['mask_{}'.format(f) for f in features]).apply(helper)


data = dict()
vec = np.array([2 ** i for i in range(n_features)])
for label in labels:
    ans = eval_auroc(label)
    kvs = np.array([[np.dot(vec, np.array(k)), v] for k, v in ans.items()])
    kvs = kvs[::-1][:, 1]
    data[label] = kvs

# color map
norm = mpl.colors.Normalize(vmin=0.7, vmax=1, clip=True)
cmap = plt.cm.viridis

# %%

import matplotlib.patches as patches

fig = plt.figure(figsize=(3, 3), dpi=300)
ax = fig.add_subplot(projection='polar')
ax.set_ylim(0, .5 + .5 + 1 * n_features + 1 * n_labels)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
fontsize = 7

'''
for missingness pattern
'''
y_start_pt = .5
y_end_pt   = .5 + 1 * n_features
x_start_pt = 4 / 2 * np.pi
x_end_pt = 1 / 2 * np.pi

# x grids
for ring in np.linspace(y_start_pt, y_end_pt, n_features + 1, endpoint=True):
    n_ticks = int(1000 * ring ** 2)
    theta = np.linspace(x_start_pt, x_end_pt, n_ticks, endpoint=False)
    ax.plot(
        theta, [ring] * n_ticks, color='black', 
        linewidth=linewidth, linestyle='-'
    )

# y_grids
for i in range(n_features):
    rings_inner = np.linspace(y_start_pt, y_end_pt, n_features + 1, endpoint=True)[i]
    rings_outer = np.linspace(y_start_pt, y_end_pt, n_features + 1, endpoint=True)[i + 1]
    for angle in np.linspace(x_start_pt, x_end_pt, 2 ** (i + 1) + 1, endpoint=True):
        ax.plot(
            [angle] * 2, [rings_inner, rings_outer], color='black', 
            linewidth=linewidth, linestyle='-'
        )

# fill between
for i in range(n_features):
    rings_inner = np.linspace(y_start_pt, y_end_pt, n_features + 1, endpoint=True)[i]
    rings_outer = np.linspace(y_start_pt, y_end_pt, n_features + 1, endpoint=True)[i + 1]
    for j in range(2 ** (i + 1)):
        angle_0 = np.linspace(x_start_pt, x_end_pt, 2 ** (i + 1) + 1, endpoint=True)[j]
        angle_1 = np.linspace(x_start_pt, x_end_pt, 2 ** (i + 1) + 1, endpoint=True)[j + 1]
        ax.fill_between(
            np.linspace(angle_0, angle_1, 1000, endpoint=True),
            [rings_inner] * 1000,
            [rings_outer] * 1000,
            facecolor = 'white',
            hatch = 'xxxx' if j % 2 == 0 else None,
            edgecolor = 'grey'
        )

'''
for performance strip
'''
y_start_pt = .5 + .5 + 1 * n_features
y_end_pt   = .5 + .5 + 1 * n_features + 1 * n_labels
x_start_pt = 4 / 2 * np.pi
x_end_pt = 1 / 2 * np.pi

# x grids
for ring in np.linspace(y_start_pt, y_end_pt, n_labels + 1, endpoint=True):
    n_ticks = int(1000 * ring ** 2)
    theta = np.linspace(x_start_pt, x_end_pt, n_ticks, endpoint=False)
    ax.plot(
        theta, [ring] * n_ticks, color='white', 
        linewidth=linewidth, linestyle='-'
    )

# y_grids
for i in range(n_labels):
    rings_inner = np.linspace(y_start_pt, y_end_pt, n_labels + 1, endpoint=True)[i]
    rings_outer = np.linspace(y_start_pt, y_end_pt, n_labels + 1, endpoint=True)[i + 1]
    for angle in np.linspace(x_start_pt, x_end_pt, 2 ** n_features + 1, endpoint=True):
        ax.plot(
            [angle] * 2, [rings_inner, rings_outer], color='white', 
            linewidth=linewidth, linestyle='-'
        )

# fill between
for i, label in enumerate(labels):
    rings_inner = np.linspace(y_start_pt, y_end_pt, n_labels + 1, endpoint=True)[i]
    rings_outer = np.linspace(y_start_pt, y_end_pt, n_labels + 1, endpoint=True)[i + 1]
    for j in range(2 ** n_features):
        angle_0 = np.linspace(x_start_pt, x_end_pt, 2 ** n_features + 1, endpoint=True)[j]
        angle_1 = np.linspace(x_start_pt, x_end_pt, 2 ** n_features + 1, endpoint=True)[j + 1]
        ax.fill_between(
            np.linspace(angle_0, angle_1, 100, endpoint=True),
            [rings_inner] * 100,
            [rings_outer] * 100,
            color = cmap(norm(data[label][j]))
        )

'''
colorbars
'''
textbox_h = 1 / (.5 + .5 + n_labels + n_features) / 2
textbox_w = textbox_h * (n_features - .5)
gap_h = textbox_h / 2

# AUROC
cbar_ax = fig.add_axes([
    0.5 + gap_h + (n_features + n_labels - 1.5) * textbox_h, 0.5 + gap_h * 2,
    gap_h, gap_h + textbox_h * (n_features + n_labels) - gap_h * 2
])
dummy_ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
dummy_ax.axis('off')
dummy_image = dummy_ax.imshow([[0, 1]], cmap=cmap, norm=norm)
dummy_image.set_visible(False)
cbar = fig.colorbar(dummy_image, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(left=True, right=False, labelleft=True, labelright=False)
cbar.ax.tick_params(direction='in')
for label in cbar.ax.yaxis.get_ticklabels():
    label.set_verticalalignment('center')
    # label.set_rotation(-90)
    # label.set_weight('bold')
    # label.set_fontsize(fontsize)
    label.set_fontsize(fontsize)

# mask
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
cbar_ax = fig.add_axes([
    0.5 + gap_h + (n_features + n_labels - 1) * textbox_h, 0.5 + gap_h * 2,
    gap_h, gap_h + textbox_h * (n_features + n_labels) - gap_h * 2
])
dummy_ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
dummy_ax.axis('off')
dummy_image = dummy_ax.imshow([[0, 1]], cmap=ListedColormap(['white', 'white']))
dummy_image.set_visible(False)
cbar = fig.colorbar(dummy_image, cax=cbar_ax, ticks=[0, 1], orientation='vertical')
cbar.ax.tick_params(direction='in')
cbar.set_ticks([.75, .25])
cbar.set_ticklabels(['Masked', 'Unmasked'])
for label in cbar.ax.yaxis.get_ticklabels():
    label.set_verticalalignment('center')
    # label.set_weight('bold')
    label.set_rotation(-90)
    label.set_fontsize(fontsize)

# Add hatching to the colorbar (in this case to the white section)
# We will use a patch to add hatching
rect = mpatches.Rectangle((0, 0.5), 1, 0.5, facecolor='white', hatch='xxxx', edgecolor='grey')
cbar.ax.add_patch(rect)
cbar.ax.set_ylim(0, 1)  # set the colorbar limits

# # Set the labels for the colorbar
# cbar.ax.set_yticklabels(['Black', 'Hatched White'])
'''
text
'''
from matplotlib.bezier import BezierSegment
ax_cartesian = fig.add_subplot(frame_on=False)  # the 111 means 1x1 grid, first subplot
ax_cartesian.axis('off')  # Turn off the Cartesian axes

'''
top text
'''
for i in range(n_labels + 1):
    # Fill between every two lines with a rectangle patch
    if i < n_labels:
        rect = patches.Rectangle(
            (.502, 1 - (i + 1) * textbox_h), textbox_w - .002, textbox_h, 
            color=list(palette_label.values())[i], alpha=1.0, transform=ax_cartesian.transAxes
        )
        ax_cartesian.add_patch(rect)

    line = plt.Line2D(
        [.5, .5 + textbox_w], [1 - i * textbox_h, 1 - i * textbox_h], color='white', 
        linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
    )
    ax_cartesian.add_line(line)

# double lines
line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w], [1, 1 - n_labels * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

line = plt.Line2D(
    [.5 + textbox_w + .005] * 2, [1, 1 - n_labels * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# text
ax_cartesian.text(
    .5 + textbox_w + gap_h, 1 - n_labels / 2 * textbox_h, 
    'AUROC', rotation=-90,
    ha='center', va='center', fontsize=fontsize,
    # weight='bold'
)
for i in range(n_labels):
    ax_cartesian.text(
        .5 + textbox_w / 2, 1 - i * textbox_h - gap_h, 
        labels[i], rotation=0,
        ha='center', va='center', fontsize=fontsize,
        # weight='bold'
    )

# corners
# a...b
# ..........c
# ..........d
# e...f
# top line
line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w + gap_h], [1, 1], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# bottom line
line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w + gap_h],
    [1 - n_labels * textbox_h, 1 - n_labels * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# right line
line = plt.Line2D(
    [.5 + textbox_h + textbox_w, .5 + textbox_h + textbox_w],
    [1 - gap_h, 1 - gap_h - (n_labels - 1) * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# top-right corner
p1 = np.array([.5 + textbox_w + gap_h, 1])
p2 = np.array([.5 + textbox_h + textbox_w, 1 - gap_h])
control = (p1 + p2) / 2 + .01
bezier = BezierSegment(np.vstack([p1, control, p2]))
cur = np.array([bezier.point_at_t(t_i) for t_i in np.linspace(0, 1, 20, endpoint=True)])
line = plt.Line2D(
    cur[:, 0], cur[:, 1], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# bottom-right corner
p1 = np.array([.5 + textbox_w + gap_h, 1 - n_labels * textbox_h])
p2 = np.array([.5 + textbox_h + textbox_w, 1 - gap_h - (n_labels - 1) * textbox_h])
control = (p1 + p2) / 2
control[0] += .01
control[1] -= .01
bezier = BezierSegment(np.vstack([p1, control, p2]))
cur = np.array([bezier.point_at_t(t_i) for t_i in np.linspace(0, 1, 20, endpoint=True)])
line = plt.Line2D(
    cur[:, 0], cur[:, 1], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

'''
bottom text
'''
for i in range(n_features + 1):
    line = plt.Line2D(
        [.5, .5 + textbox_w], [.5 + gap_h + i * textbox_h, .5 + gap_h + i * textbox_h], color='grey', 
        linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
    )
    ax_cartesian.add_line(line)

line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w], [.5 + gap_h, .5 + gap_h + n_features * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)
line = plt.Line2D(
    [.5 + textbox_w + .005] * 2, [.5 + gap_h, .5 + gap_h + n_features * textbox_h], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# top line
line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w + gap_h], 
    [.5 + gap_h + textbox_h * n_features] * 2, color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# bottom line
line = plt.Line2D(
    [.5 + textbox_w, .5 + textbox_w + gap_h],
    [.5 + gap_h] * 2, color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# right line
line = plt.Line2D(
    [.5 + textbox_h + textbox_w] * 2,
    [.5 + textbox_h, .5 + textbox_h * n_features], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# top-right corner
p1 = np.array([.5 + textbox_w + gap_h, .5 + textbox_h * n_features + gap_h])
p2 = np.array([.5 + textbox_h + textbox_w, .5 + textbox_h * n_features])
control = (p1 + p2) / 2 + .01
bezier = BezierSegment(np.vstack([p1, control, p2]))
cur = np.array([bezier.point_at_t(t_i) for t_i in np.linspace(0, 1, 20, endpoint=True)])
line = plt.Line2D(
    cur[:, 0], cur[:, 1], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# bottom-right corner
p1 = np.array([.5 + textbox_w + gap_h, .5 + gap_h])
p2 = np.array([.5 + textbox_h + textbox_w, .5 + textbox_h])
control = (p1 + p2) / 2
control[0] += .01
control[1] -= .01
bezier = BezierSegment(np.vstack([p1, control, p2]))
cur = np.array([bezier.point_at_t(t_i) for t_i in np.linspace(0, 1, 20, endpoint=True)])
line = plt.Line2D(
    cur[:, 0], cur[:, 1], color='grey', 
    linewidth=linewidth, linestyle='-', transform=ax_cartesian.transData
)
ax_cartesian.add_line(line)

# text
ax_cartesian.text(
    .5 + textbox_w + gap_h, 0.5 + gap_h + n_features / 2 * textbox_h, 
    'Feature masks', rotation=-90,
    ha='center', va='center', fontsize=fontsize,
    # weight='bold',
)
for i in range(n_features):
    ax_cartesian.text(
        .5 + textbox_w / 2, .5 + i * textbox_h + gap_h * 2, 
        feature_displays[features[i]], rotation=0,
        ha='center', va='center', fontsize=fontsize,
        # weight='bold',
    )

# plt.savefig("figure_2c.png", transparent=False)
plt.savefig("final_figs/figure_2a.pdf", dpi=300)



# %%