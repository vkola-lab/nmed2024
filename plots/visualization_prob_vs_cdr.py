
# %%
import matplotlib.font_manager
fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
print(fonts)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
import scipy.stats as stats
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
import matplotlib
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


from matplotlib import rc, rcParams
rc('axes', linewidth=0.5)
rc('font', size=7)
rcParams['font.family'] = 'Arial'

basedir = '../model_predictions_stripped_MNI_swinunetr'
df1 = pd.read_csv(f'{basedir}/nacc_test_with_np_cli_swinunetr_prob.csv')
df2 = pd.read_csv(f'{basedir}/adni_merged_swinunetr_prob.csv')
df3 = pd.read_csv(f'{basedir}/fhs_converted_6months_cleaned_swinunetr_prob.csv')
df3['cdr_CDRGLOB'] = df3['cdr_CDRGLOB'].replace({2.0:1.0, 3.0:1.0}).astype(float)

df1.cdr_CDRGLOB = df1.cdr_CDRGLOB.astype('str')
df2.cdr_CDRGLOB = df2.cdr_CDRGLOB.astype('str')
df3.cdr_CDRGLOB = df3.cdr_CDRGLOB.astype('str')

# %%
def normality_test(df):
    for g in df['cdr_CDRGLOB'].unique():
        group = df[df['cdr_CDRGLOB'] == g]['DE_prob']
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

print('NACC')
normality_test(df1)
print('ADNI')
normality_test(df2)
print('FHS')
normality_test(df3)
print()
print(stats.shapiro(df1.DE_prob).pvalue)
print(stats.shapiro(df2.DE_prob).pvalue)
print(stats.shapiro(df3.DE_prob).pvalue)    

# %%

def get_kruskal_dunn_pvalues(df, pairs):
    
    group_labels = sorted(list(set([value for pair in pairs for value in pair])))
    
    data = [
        list(df[df['cdr_CDRGLOB'] == cdr]['DE_prob']) for cdr in group_labels
    ]
    
    for i in data:
        print(len(i))


    # Flatten the 'data' list to a single list of values with corresponding group labels
    values = []
    for i, group_data in enumerate(data):
        values.extend([(value, group_labels[i]) for value in group_data])

    # Create a DataFrame from the flattened data
    df_ = pd.DataFrame(values, columns=['Value', 'Group'])
    
    print("n: ", len(df_))
    
    for pair in pairs:
        # print(pair)
        print(pair, len(df[(df['cdr_CDRGLOB'] == pair[0]) | (df['cdr_CDRGLOB'] == pair[1])]))
    
    # Perform the Kruskal-Wallis test
    kw_result = stats.kruskal(*[group_data for label, group_data in df_.groupby('Group')['Value']])

    # Print the Kruskal-Wallis test result
    print("Kruskal-Wallis Test:")
    print(f"Statistically significant: {kw_result.pvalue < 0.05}", kw_result.pvalue, kw_result)

    # Perform post hoc pairwise comparisons using Dunn's test
    posthoc_dunn = sp.posthoc_dunn(df_, val_col='Value', group_col='Group', p_adjust='bonferroni')
    # print(posthoc_dunn)
    # print(posthoc_dunn[posthoc_dunn.columns[::-1]])
    posthoc_dunn = posthoc_dunn[posthoc_dunn.columns[::-1]]
    p_values = {val: dict(posthoc_dunn[val]) for val in group_labels}
    
    values = [p_values[i][j] for (i,j) in pairs]
    
    return posthoc_dunn, values
    # return values

ax1_pairs = [('0.0', '0.5'), ('0.0', '1.0'), ('0.0', '2.0'), ('0.0', '3.0'), ('0.5', '1.0'), ('0.5', '2.0'), ('0.5', '3.0'), ('1.0', '2.0'), ('1.0', '3.0'), ('2.0', '3.0')]
ax2_pairs = [('0.0', '0.5'), ('0.5', '1.0'), ('0.0', '1.0')]
print(get_kruskal_dunn_pvalues(df1, ax1_pairs))

print(get_kruskal_dunn_pvalues(df2, ax2_pairs))

print(get_kruskal_dunn_pvalues(df3, ax2_pairs))


# %%
order = ['0.0', '0.5', '1.0', '2.0', '3.0']
colors = [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0), (216/255, 27/255, 96/255, 1.0), (216/255, 27/255, 96/255, 1.0)]

def generate_plot(df, ax, cohort, order, bp_width=0.5, strip_sz=1, strip_jitter=0.24, figu="Figure 2a", xlabel='Clinical Dementia Rating', xticks=None):
    bp = sns.boxplot(data=df, x='cdr_CDRGLOB', y='DE_prob', order=order, ax=ax, showfliers=False, width=bp_width, linewidth=0.5) #, gridspec_kw={'width_ratios': [1.3, 1]})
    for ind, box in enumerate(bp.patches):
        box.set_facecolor('none')
        box.set_edgecolor(colors[ind])
        whiskers = bp.lines[5 * ind:5 * (ind+1)]
        # print(whiskers)
        for whisker in whiskers:
            whisker.set_color(colors[ind])
        # break
        
    # Violinplot
    vp = sns.violinplot(
        data=df,
        x="cdr_CDRGLOB", y="DE_prob", 
        order=order, hue=True,
        hue_order=[True, False], split=True,
        ax=ax, linewidth=0.5
    )
    ax.legend([],[], frameon=False)
    for ind, violin in enumerate(vp.findobj(PolyCollection)):
        violin.set_facecolor(colors[ind])
        violin.set_alpha(0.4) 


    # Stripplot
    sns.stripplot(data=df, x='cdr_CDRGLOB', y='DE_prob', order=order, palette=colors, ax=ax, size=strip_sz, jitter=strip_jitter, edgecolor='black')
    
    ax.set(
        xlabel=xlabel, 
        ylabel='$P_{DE}$',
        title=cohort
    )
    
    
    ax.set_yticks(ax.get_yticks()[(ax.get_yticks() >= 0.0) & (ax.get_yticks() <= 1.2)])
    
    if xticks:
        ax.set_xticks([0.0, 1.0, 2.0], xticks)



def map_values(value):
    if isinstance(value, str):
        return value
    if value < 0.0001:
        return '**** '
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
    # print(matrix_content_values)

    matrix_with_indices = np.column_stack((matrix_content_values, row_indices))
    column_indices = np.insert(column_indices.values, 0, '', axis=0)
    matrix_with_indices = np.row_stack((matrix_with_indices, column_indices))
    print(matrix_with_indices)
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
        
    main[-1] = main[-1] + "        "
    main = main[1:]
    return main

def annotate(df, ax, pairs, order, cohort, xy=(0.95, 0.03)):
    annot = Annotator(ax, pairs=pairs, data=df, x='cdr_CDRGLOB', y='DE_prob', order=order)
    matrix_content, _ = get_kruskal_dunn_pvalues(df, pairs=pairs)
    matrix_text = get_annotate_matrix(matrix_content)
    
    
    if cohort in ['NACC']:
        matrix_text.insert(0, 'p-values{: <14}'.format(' '))
    elif cohort in ['ADNI']:
        matrix_text.insert(0, 'p-values{: <5}'.format(' '))
    else:
        for i, li in enumerate(matrix_text):
            matrix_text[i] = li.replace('0.0', 'Norm').replace('0.5', 'Imp').replace('1.0', 'Dem')
        matrix_text.insert(0, 'p-values{: <8}'.format(' '))
    matrix_text = '\n'.join(matrix_text)
    print(matrix_text)
    ax.annotate(matrix_text, xy=xy, xycoords='axes fraction', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=(1, 1, 1, 0.0), lw=0.5), fontsize=7)


ax1_pairs = [('0.0', '0.5'), ('0.0', '1.0'), ('0.0', '2.0'), ('0.0', '3.0'), ('0.5', '1.0'), ('0.5', '2.0'), ('0.5', '3.0'), ('1.0', '2.0'), ('1.0', '3.0'), ('2.0', '3.0')]
ax2_pairs = [('0.0', '0.5'), ('0.5', '1.0'), ('0.0', '1.0')]



#%%
fig = plt.figure(figsize=(2.3, 2.3), dpi=300)
ax = fig.add_subplot(111)
generate_plot(df=df1, ax=ax, order=order, cohort='NACC', bp_width=0.42, strip_sz=0.3, strip_jitter=0.20, figu="Figure 2i")
annotate(df=df1, ax=ax, pairs=ax1_pairs, order=order, cohort='NACC', xy=(0.97, 0.03))
# plt.legend(fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig('final_figs/fig_cdr_nacc.pdf', format='pdf', dpi=300, bbox_inches='tight')

#%%
fig = plt.figure(figsize=(2.3, 2.3), dpi=300)
ax = fig.add_subplot(111)
generate_plot(df=df2, ax=ax, order=order[:-2], cohort='ADNI', bp_width=0.26, strip_sz=0.5, strip_jitter=0.13, figu="Figure 2j")
annotate(df=df2, ax=ax, pairs=ax2_pairs, order=order[:-2], cohort='ADNI', xy=(0.415, 0.718))
# plt.legend(fontsize=18)
plt.tight_layout()
# plt.show()
plt.savefig('final_figs/fig_cdr_adni.pdf', format='pdf', dpi=300, bbox_inches='tight')

#%%
fig = plt.figure(figsize=(2.3, 2.3), dpi=300)
ax = fig.add_subplot(111)
generate_plot(df=df3, ax=ax, order=order[:-2], cohort='FHS', bp_width=0.26, strip_sz=0.5, strip_jitter=0.13, figu="Figure 2k", xlabel='             ',xticks=['Normal', 'Imp', 'Dementia'])
annotate(df=df3, ax=ax, pairs=ax2_pairs, order=order[:-2], cohort='FHS', xy=(0.485, 0.718))
# plt.legend(fontsize=18)
plt.tight_layout()
# plt.show()
plt.savefig('final_figs/fig_cdr_fhs.pdf', format='pdf', dpi=300, bbox_inches='tight')


# %%


