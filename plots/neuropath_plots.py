#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:22:36 2023

@author: meaha
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_dunn
from scikit_posthocs import posthoc_tukey
from statannotations.Annotator import Annotator
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
import copy
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
from scipy.stats import ks_2samp, mannwhitneyu

from matplotlib import rc, rcParams

rc('axes', linewidth=0.5)
rc('font', size=7)
rcParams['font.family'] = 'Arial'
fontsize = 7



#%%
## boxplot with ANOVA and tukey test
pval_thresh = 0.05 #pval>threshold is not shown on the tukey annotation


NP_feature = "NACCAMY"
NP_feature_label = 'Cerebral amyloid angiopathy'
other_subtype = "AD probability"  #dementia subtype etiology
other_subtype_label = "AD"
legend_loc = 4
label_vals = [0,1]
label_names = ["None", "Mild/Moderate/Severe"]

# NP_feature = "NPTHAL"
# NP_feature_label = 'A score'
# other_subtype = "AD probability"  #dementia subtype etiology
# other_subtype_label = "AD"
# legend_loc = 4
# label_vals = [0,1]
# label_names = ["Phase A0", "Phase A1-A3"]
################# NACC other subtype
df_nacc_other = pd.read_csv(
    "np_data/nacc_DE.csv", 
    usecols=[other_subtype,NP_feature])
df_nacc_other[NP_feature] = df_nacc_other[NP_feature].map({0:0, 1:1, 2:1, 3:1, 4:1, 5:1})
# df_nacc_other.loc[df_nacc_other[NP_feature] == 0, NP_feature] = np.NaN
# df_nacc_other.loc[df_nacc_other[NP_feature] == 3, NP_feature] = np.NaN


df_nacc_other = df_nacc_other.dropna()
NP_nacc_other = np.unique(df_nacc_other[NP_feature])
data_nacc_other = []

for s in NP_nacc_other:
    data_nacc_other.append(df_nacc_other[df_nacc_other[NP_feature] == s][other_subtype])
# data_nacc_other.pop()  #remove last index which is null
#%%
################### ADNI other subtype
df_adni_other = pd.read_csv(
    "np_data/adni_DE.csv",
    usecols=[other_subtype,NP_feature])
df_adni_other[NP_feature] = df_adni_other[NP_feature].map({0:0, 1:1, 2:1, 3:1, 4:1, 5:1})
df_adni_other = df_adni_other.dropna()
NP_adni_other = np.unique(df_adni_other[NP_feature])
data_adni_other = []


for a in NP_adni_other:
    data_adni_other.append(df_adni_other[df_adni_other[NP_feature] == a][other_subtype])
    

################### FHS other subtype
df_fhs_other = pd.read_csv(
    "np_data/fhs_DE.csv",
    usecols=[other_subtype,NP_feature])
df_fhs_other[NP_feature] = df_fhs_other[NP_feature].map({0:0, 1:1, 2:1, 3:1, 4:1, 5:1})
df_fhs_other = df_fhs_other.dropna()
NP_fhs_other = np.unique(df_fhs_other[NP_feature])
data_fhs_other = []

for a in NP_fhs_other:
    data_fhs_other.append(df_fhs_other[df_fhs_other[NP_feature] == a][other_subtype])
        
#%%
############################# merge datasets other subtypes
df_concat_other = pd.concat([df_fhs_other, df_nacc_other, df_adni_other], axis=0)
# df_concat_other = pd.concat([df_nacc_other], axis=0)


dfl_nacc_other = pd.melt(df_nacc_other, id_vars=other_subtype, value_vars=[NP_feature])
dfl_adni_other = pd.melt(df_adni_other, id_vars=other_subtype, value_vars=[NP_feature])
dfl_fhs_other = pd.melt(df_fhs_other, id_vars=other_subtype, value_vars=[NP_feature])

dfl_nacc_rows_other = len(dfl_nacc_other.index)
dfl_adni_rows_other = len(dfl_adni_other.index)
dfl_fhs_rows_other = len(dfl_fhs_other.index)


list_nacc_other = ["NACC" for x in range(dfl_nacc_rows_other)]
list_adni_other = ["ADNI" for x in range(dfl_adni_rows_other)]
list_fhs_other = ["FHS" for x in range(dfl_fhs_rows_other)]

list_nacc_other_subtype = [other_subtype for x in range(dfl_nacc_rows_other)]
list_adni_other_subtype = [other_subtype for x in range(dfl_adni_rows_other)]
list_fhs_other_subtype = [other_subtype for x in range(dfl_fhs_rows_other)]


dfl_nacc_other['Dataset'] = list_nacc_other
dfl_adni_other['Dataset'] = list_adni_other
dfl_fhs_other['Dataset'] = list_fhs_other

dfl_nacc_other['Etiology'] = list_nacc_other_subtype
dfl_adni_other['Etiology'] = list_adni_other_subtype
dfl_fhs_other['Etiology'] = list_fhs_other_subtype

dfl_concat_other = pd.concat([dfl_fhs_other, dfl_nacc_other, dfl_adni_other], axis=0) #all datasets

# dfl_concat_other = pd.concat([dfl_nacc_other], axis=0) #all datasets

#%%
def normality_test(df):
    for g in df['value'].unique():
        group = df[df['value'] == g]['AD probability']
        if len(group) > 3: 
            try:
                # Shapiro-Wilk test
                shapiro_pvalue = stats.shapiro(group).pvalue
                print(f'Group {g}: Shapiro-Wilk Test, P-value = {shapiro_pvalue}, Normality Rejected: {shapiro_pvalue < 0.05}, Sample Size: {len(group)}')
                
                # # For K-S test, standardizing or adjusting parameters
                # ks_pvalue = stats.kstest(group, 'norm', args=(group.mean(), group.std())).pvalue
                # print(f'Group {g}: K-S Test, P-value = {ks_pvalue}, Normality Rejected: {ks_pvalue < 0.05}')
                
            except Exception as e:
                print(f'Error testing group {g}: {e}')
                continue

print('NACC')
normality_test(dfl_concat_other)
print()
# print(stats.shapiro(df1.DE_prob).pvalue)
# print(stats.shapiro(df2.DE_prob).pvalue)
# print(stats.shapiro(df3.DE_prob).pvalue)  

#%%
############## Fix legends ##############
from matplotlib.axes._axes import Axes
from matplotlib.markers import MarkerStyle
from seaborn import color_palette
from numpy import ndarray

def GetColor2Marker(markers):
    palette = color_palette()
    # palette = ['orange','blue','green','red']
    mkcolors = [(palette[i]) for i in range(len(markers))]
    return dict(zip(mkcolors,markers))

def fixlegend(ax,markers,markersize=8,**kwargs):
    # Fix Legend
    legtitle =  ax.get_legend().get_title().get_text()
    _,l = ax.get_legend_handles_labels()
    palette = color_palette()

    mkcolors = [(palette[i]) for i in range(len(markers))]
    newHandles = [plt.Line2D([0],[0], ls="none", marker=m, color=c, mec="none", markersize=markersize,**kwargs) \
                for m,c in zip(markers, mkcolors)]
    ax.legend(newHandles,l)
    leg = ax.get_legend()
    leg.set_title(legtitle)

old_scatter = Axes.scatter
def new_scatter(self, *args, **kwargs):
    colors = kwargs.get("c", None)
    co2mk = kwargs.pop("co2mk",None)
    FinalCollection = old_scatter(self, *args, **kwargs)
    if co2mk is not None and isinstance(colors, ndarray):
        Color2Marker = GetColor2Marker(co2mk)
        paths=[]
        for col in colors:
            mk=Color2Marker[tuple(col)]
            marker_obj = MarkerStyle(mk)
            paths.append(marker_obj.get_path().transformed(marker_obj.get_transform()))
        FinalCollection.set_paths(paths)
    return FinalCollection
Axes.scatter = new_scatter
############## End hack. ##############
#%%

# plt.legend(title="NP scores", fontsize=30, title_fontsize=35, 
#             loc=4, frameon=True, ncol=1, shadow=True)

# palette = ['green', 'blue', 'orange', 'red', 'magenta', 'cyan', 'yellow']

plt.rcParams["figure.figsize"] = (2.3,2.3)
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize)

#Nature Medicine version
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# plt.rc('xtick', labelsize=7) 
# plt.rc('ytick', labelsize=7)




# colors = sns.color_palette('Set1')
Markers = ["^","o", "d", "s"]

sw = sns.swarmplot(x='value', y=other_subtype, 
              data=dfl_concat_other, hue='Dataset', 
                dodge=True, co2mk=Markers, size=2.5)

#change color manually
# sw = sns.swarmplot(x='value', y=other_subtype, 
#               data=dfl_concat_other, hue='Dataset', 
#                 dodge=True, marker=Markers[1], size=40, palette=['darkorange'])

fixlegend(sw,Markers,markersize=1)

## binary scores
# vp = sns.violinplot(x='value', y=other_subtype, data=dfl_concat_other, 
#                     linewidth=2, palette=["y", "lightpink", "paleturquoise",
#                                           "thistle", "sandybrown", 
#                                           "darkkhaki", "palegreen"], 
#                     saturation=0.75, width=.8)


## non binary scores
vp = sns.violinplot(x='value', y=other_subtype, data=dfl_concat_other, 
                    linewidth=0.5, palette=["y",  "lightpink", "paleturquoise",
                                           "thistle", "sandybrown", 
                                          "darkkhaki", "palegreen"], 
                    saturation=0.75) #, width=.8)

for ind, violin in enumerate(vp.findobj(PolyCollection)):
    # violin.set_facecolor(colors[ind])
    violin.set_alpha(0.5) 

# bp_all = sns.boxplot(x='value', y=DE_subtype, data=dfl_concat_all, showfliers=False, 
#               hue='variable', width=0.3,
#             linewidth=3, color='white')

bp = sns.boxplot(x='value', y=other_subtype, data=dfl_concat_other, showfliers=False, 
              width=0.25, linewidth=0.5, 
              boxprops=dict(alpha=1), color="white", fliersize=0.5)

for ind, box in enumerate(bp.patches):
    whiskers = bp.lines[5 * ind:5 * (ind+1)]
    # print(whiskers)
    for whisker in whiskers:
        whisker.set_linewidth(0.5)
plt.legend().set_visible(False)

################# Sahana code for tukey test box ########################
def map_values(value):
    if isinstance(value, str):
        return value
    if value < 0.0001:
        return '****'
    elif value < 0.001:
        return '***'
    elif value < 0.01:
        return '**'
    elif value < 0.05:
        return '*'
    elif value <= 1.0:
        return 'ns'
    else:
        return str(value)
    
def get_annotate_matrix(matrix_content):
    print(matrix_content)
    row_indices = matrix_content.index
    column_indices = matrix_content.columns

    matrix_content_values = np.vectorize(map_values)(matrix_content.values)
    # matrix_content_values = np.triu(matrix_content_values)
    # print(matrix_content_values)

    # matrix_with_indices = matrix_content.values
    matrix_with_indices = np.column_stack((matrix_content_values, row_indices))
    print(matrix_with_indices)
    column_indices = np.insert(column_indices.values, 0, '', axis=0)
    matrix_with_indices = np.row_stack((matrix_with_indices, column_indices))
    print(matrix_with_indices)
    # matrix_text = '\n'.join(['  '.join('{: <4}'.format(str(cell)) for cell in row) for i, row in enumerate(matrix_with_indices)])
    main = []
    # main.append('p-values              ')
    for i, row in enumerate(matrix_with_indices):
        if i == len(matrix_with_indices) - 1:
            row = row[2:]
        elif i == 0:
            row = row[len(row) - i:]
        else:
            row = row[len(row) - i - 1:]
        cr = []
        for cell in row:
            cr.append('{: <5}'.format(str(cell)))
        main.append('  '.join(cr))
        
    main[-1] = main[-1] + "         "
    main = main[1:]
    return main
# order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0']
def annotate(df_Tukey, plt, tukey_df, xy, etiology):
    annot = Annotator(
        bp, pairs_other, data=df_Tukey, x=NP_feature, y=etiology, line_width=0.5)
    matrix_text = get_annotate_matrix(tukey_df)
    matrix_text = '\n'.join(matrix_text)
    plt.annotate(matrix_text, xy=xy, xycoords='axes fraction', fontsize=fontsize, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=(1, 1, 1, 0.0)))


##################################### Tukey test other subtype
## box annotation

v1, v2 = sorted(df_concat_other[NP_feature].unique())
smp1 = df_concat_other[df_concat_other[NP_feature] == v1][other_subtype]
smp2 = df_concat_other[df_concat_other[NP_feature] == v2][other_subtype]
statistic, p_value = mannwhitneyu(smp1, smp2, alternative='less')
print("mann whitney", mannwhitneyu(smp1, smp2, alternative='less'))
p_values_other = [p_value]
pairs_other = [(v1, v2)]

##line post-hoc annotation
if len(p_values_other) != 0:
    ns_idxs_other = []
    for i_other in range(len(p_values_other)):
        if p_values_other[i_other] > pval_thresh:
            ns_idxs_other.append(i_other)
    # important to delete not significant p-values        
    for index_other in sorted(ns_idxs_other, reverse=True):
        del pairs_other[index_other]
        del p_values_other[index_other]
        
    #text_format = "simple" or "star"
    annotator = Annotator(
        bp, pairs_other, data=df_concat_other, x=NP_feature, y=other_subtype, line_width=0.5)
    annotator.configure(text_format="star", loc="inside", 
                          fontsize=9, color='black', use_fixed_offset=False, line_width=0.5)
    annotator.set_pvalues_and_annotate(p_values_other)
    annotator.print_pvalue_legend()
    plt.tight_layout()


plt.ylabel(other_subtype, fontsize=fontsize, fontname="Arial")
plt.xlabel(NP_feature_label, fontsize=fontsize, fontname="Arial")
plt.tick_params(axis='both', which='major')
plt.tick_params(labelsize=fontsize)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set(lw=0.5)
plt.gca().spines['bottom'].set(lw=0.5)
plt.xticks(fontname = "Arial")
plt.yticks(fontname = "Arial")


#manual label for x pivot
plt.xticks(label_vals, label_names, fontname = "Arial")
plt.tick_params(axis='both', which='both')

#Nature Medicine version
# plt.tick_params(axis='both', which='both', length=10)

# plt.ylim((0,1))
file_name = './final_figs/'+NP_feature+"_"+other_subtype_label+"_new.pdf"
plt.savefig(file_name, format='PDF', dpi=300, bbox_inches='tight')

#######################################################
# %%
