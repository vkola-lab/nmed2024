#%% load data and compute metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statannotations.Annotator import Annotator
from scipy.stats import ks_2samp
from matplotlib import rc, rcParams
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, recall_score
from matplotlib.collections import PolyCollection
import scipy.stats as stats

rc('axes', linewidth=0.5)
rc('font', size=7)
rcParams['font.family'] = 'Arial'


color_scheme = dict(zip(
    ['NC', 'MCI', 'DE'],
    [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
))
cb_palette = sns.color_palette('colorblind')



#%%
df_pred = pd.read_csv('/home/skowshik/publication_ADRD_repo/adrd_tool/model_predictions_stripped_MNI_swinunetr/nacc_test_with_np_cli_swinunetr_prob.csv')
df_raw = pd.read_csv('/data_1/skowshik/data_files/nacc/new_nacc_unique_type_3.csv')

# %%
def normality_test(df, variable, values):
    for g in values:
        group = df[df['NACCALZP'] == g][variable]
        if len(group) > 3: 
            try:
                # Shapiro-Wilk test
                shapiro_pvalue = stats.shapiro(group).pvalue
                print(f'Group {g}: Shapiro-Wilk Test, P-value = {shapiro_pvalue}, Normality Rejected: {shapiro_pvalue < 0.05}, Sample Size: {len(group)}')
                
                # For K-S test, standardizing or adjusting parameters
                ks_pvalue = stats.kstest(group, 'norm', args=(group.mean(), group.std())).pvalue
                print(f'Group {g}: K-S Test, P-value = {ks_pvalue}, Normality Rejected: {ks_pvalue < 0.05}')
                
            except Exception as e:
                print(f'Error testing group {g}: {e}')
                continue


#%%
# colors = [(30/255, 136/255, 229/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
colors = [cb_palette[0], cb_palette[3]]
df_merged = pd.merge(df_pred, df_raw, on=['ID', 'cdr_CDRGLOB'], how='left')
df1 = df_merged[df_merged['MCI_label'] == 1]
df1['et'] = 'MCI'
df2 = df_merged[df_merged['DE_label'] == 1]
df2['et'] = 'DE'
df_total = pd.concat([df1, df2])

#%%
df1['NACCALZP'] = df1['NACCALZP'].replace({
    1: 13,
    2: 13,
    3: np.NaN
})

df2['NACCALZP'] = df2['NACCALZP'].replace({
    1: 13,
    2: 13,
    3: np.NaN
})

# compute stats
pairs = [
    (13, 7),
    # (1, 7),
    # (2, 7)
]

ps = []
for v1, v2 in pairs:
    smp1 = df1[df1['NACCALZP'] == v1]['AD_prob']
    smp2 = df1[df1['NACCALZP'] == v2]['AD_prob']
    statistic, p_value = ks_2samp(smp1, smp2)
    print(len(df1), len(smp1), len(smp2))
    print(statistic, p_value)
    print()
    # ps.append(str(p_value))
    ps.append(p_value)
    
for v1, v2 in pairs:
    smp1 = df2[df2['NACCALZP'] == v1]['AD_prob']
    smp2 = df2[df2['NACCALZP'] == v2]['AD_prob']
    statistic, p_value = ks_2samp(smp1, smp2)
    print(len(df2), len(smp1), len(smp2))
    print(statistic, p_value)
    # ps.append(str(p_value))
    ps.append(p_value)
    
# plotting_parameters = {
#     'data': df_total,
#     'y': 'AD_prob',
#     'x': 'NACCALZP',
# }

#%%
import ptitprince1 as pt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np

df_total['NACCALZP'] = df_total['NACCALZP'].replace({
    1: 1,
    2: 1,
    7: 0,
    3: np.NaN
})

#%%
normality_test(df_total[df_total['MCI_label'] == 1], 'AD_prob', [0,1])
normality_test(df_total[df_total['DE_label'] == 1], 'AD_prob', [0,1])

#%%

set2_palette = sns.color_palette("Set3")
custom_pal = {0: set2_palette[2], 1: set2_palette[3]}


dx = "et"; dy = "AD_prob"; dhue = "NACCALZP"; ort='v'
title_size = 7
label_fontsize = 7
ticks_fontsize = 7
legend_fontsize = 7

f, ax = plt.subplots(figsize=(4.6, 2.3), dpi=300)
# RainCloud plot
ax = pt.RainCloud(x=dx, y=dy, hue=dhue, data=df_total, palette=custom_pal, bw=0.2, linewidth = 0.5, width_viol=.7, ax=ax, orient=ort, alpha=.65, dodge=True, cut = 2, point_size = 0.5, jitter=0.08)

current_left, current_right = ax.get_xlim()
ax.set_xlim(current_left - 0.55, current_right + 0.06)

# add stats
x_pos = [-0.05, 0.06]  # X positions where you want the annotation to span
y_pos = df_total[dy].max() + 0.03  # Y position, slightly above the highest data point or box plot
ax.plot(x_pos, [y_pos, y_pos], color="black", lw=0.5)  # Draw the line between the points we're comparing
ax.text(np.mean(x_pos), y_pos , '**', ha='center', va='bottom', color="black", fontsize=6) # Add asterisk for significance

x_pos = [0.95, 1.06]  # X positions where you want the annotation to span
y_pos = df_total[dy].max() + 0.03  # Y position, slightly above the highest data point or box plot
ax.plot(x_pos, [y_pos, y_pos], color="black", lw=0.5)  # Draw the line between the points we're comparing
ax.text(np.mean(x_pos), y_pos , '****', ha='center', va='bottom', color="black", fontsize=6)  # Add asterisk for significance

new_legend_labels = ['No AD', 'AD as primary\nor contributing\nfactor']
# get unique labels to avoid repetitions for each distribution
handles, original_labels = ax.get_legend_handles_labels()
# rename legend and reposition
legend = ax.legend(handles, new_legend_labels,
                loc='upper left', bbox_to_anchor=(0, 1), fontsize = 6)

# rename and resize labels
ax.set_xlabel("", fontsize = 7)
ax.set_ylabel("$P_{AD}$", fontsize = 7)
ax.set_xticklabels(['MCI', 'DE'], fontsize = 7)
# ax.set_yticklabels(np.arange(0, 1.1, 0.2).astype(str), fontsize = 7)
yticks = np.arange(0, 1.1, 0.2)
ax.set_yticks(yticks)
ax.tick_params(axis='y', labelsize=7)
# plt.title("Model Predicted AD Probabilities by Amyloid Status", fontsize=16)
# plt.setp(ax.get_legend().get_texts(), fontsize=28) # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize=28) # for legend title
plt.subplots_adjust(right=0.75)
# plt.show()
plt.savefig("final_figs/fig_mci_ad_noad.pdf", format='pdf', dpi=300, bbox_inches='tight')
