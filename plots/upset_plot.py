
# %%
import pandas as pd
from UpSetPlot.upsetplot import plot
from matplotlib import pyplot as plt
from UpSetPlot.upsetplot import generate_counts, plot
import matplotlib.patches as mpatches
from UpSetPlot.upsetplot import UpSet
from UpSetPlot.upsetplot import from_contents
from matplotlib import cm
import numpy as np


df = pd.read_csv('../model_predictions_stripped_MNI_swinunetr/nacc_test_with_np_cli_swinunetr_prob.csv')
disease_list = ['AD', 'LBD', 'VD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'PRD', 'ODE']
labels = [f'{et}_label' for et in disease_list]

sets = []
for label in labels:
    tmp_df = df[df[label] == 1]
    sets.append(set(tmp_df['ID'].values))

df = df.set_index(df['ID'])

def get_avg_prob(row):
    one_cols = [col for col in labels if row[col] == 1]
    zero_cols = [col for col in labels if row[col] == 0]
    
    # one_prob_cols = [col.split('_')[0] + '_prob' for col in one_cols]
    # # print(prob_cols)
    # if len(one_prob_cols) == 0:
    #     return row['DE_prob']
    # return row[one_prob_cols].prod()
    
    one_prob_cols = list(row[[col.split('_')[0] + '_prob' for col in one_cols]])
    zero_prob_cols = list(1 - row[[col.split('_')[0] + '_prob' for col in zero_cols]])
    # one_prob_cols = [np.log10(val) for val in one_prob_cols]
    # zero_prob_cols = [np.log10(val) for val in zero_prob_cols]
    prob_cols = np.array(one_prob_cols + zero_prob_cols)
    return np.log10(np.prod(prob_cols))
    # return np.sum(prob_cols)

df['prob'] = df.apply(get_avg_prob, axis=1)
df.to_csv('check.csv', index=False)
df
    
#%%
from matplotlib.colors import Colormap, to_hex
import matplotlib
import numpy as np
num_colors = 10

# Get the "viridis" colormap
cmap = matplotlib.colormaps["viridis"]
cmap_idx_list = [int(i) for i in np.linspace(0, cmap.N, num_colors)]
colors = [to_hex(cmap(i), keep_alpha=True) for i in cmap_idx_list]
print(colors)


from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', size=40)
plt.rcParams['font.family'] = 'Arial'

fig = plt.figure(figsize=(50, 65), dpi=300)
all_elems = set(df['ID'].values)

plot_df = pd.DataFrame([[e in s for s in sets] for e in all_elems], index=list(all_elems), columns=labels)
plot_df = plot_df.rename(columns=lambda x: x + '>')
concat_df = pd.concat([df, plot_df], axis=1)
concat_df = concat_df.set_index(list(plot_df.columns))
concat_df.columns
upset = UpSet(concat_df, totals_plot_elements=6, intersection_plot_elements=8, max_subset_size=None, min_subset_size=2, max_degree=6, element_size=None, subset_size='auto', orientation='Verticle', show_counts=False, sort_by='-degree', sort_categories_by='cardinality')#, shading_color='lightgrey')

upset.style_subsets(min_subset_size=100, max_subset_size=None, facecolor=colors[0])
upset.style_subsets(min_subset_size=20, max_subset_size=99, facecolor=colors[2])
upset.style_subsets(min_subset_size=10, max_subset_size=19, facecolor=colors[4])
upset.style_subsets(min_subset_size=4, max_subset_size=9, facecolor=colors[6])
upset.style_subsets(min_subset_size=None, max_subset_size=3, facecolor=colors[8])

upset.add_catplot(value='prob', kind='box', elements=7, saturation=1, flierprops={"marker": "o"}) #, palette='viridis') #, size=3)
upset.plot(fig=fig)

plt.savefig('final_figs/upsetplot.pdf', format='pdf', dpi=300, bbox_inches='tight')


# %%
