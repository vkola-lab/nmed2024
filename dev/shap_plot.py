# %%
import shap
import numpy as np
import matplotlib.pyplot as plt
import textwrap

import pandas as pd
df = pd.read_csv('/home/skowshik/shap_results_031224.csv')
# df = pd.read_csv('./shap_results_121023.csv')

# image column names
col_names  = [f'img_MRI_T1_{i}' for i in range(1, 11)]
col_names += [f'img_MRI_T2_{i}' for i in range(1, 8)]
col_names += [f'img_MRI_FLAIR_{i}' for i in range(1, 8)]
col_names += [f'img_MRI_SWI_{i}' for i in range(1, 2)]

# drop rows where no images are available
# df = df[~(df[col_names] == 0).all(axis=1)]

# average SHAP values of ima_MRI_x
df['Brain MRI (T1)'] = df[[f'img_MRI_T1_{i}' for i in range(1, 11)]].sum(axis=1)
df['Brain MRI (T2)'] = df[[f'img_MRI_T2_{i}' for i in range(1, 8)]].sum(axis=1)
df['Brain MRI (FLAIR)'] = df[[f'img_MRI_FLAIR_{i}' for i in range(1, 8)]].sum(axis=1)
df['Brain MRI (SWI)'] = df[[f'img_MRI_SWI_{i}' for i in range(1, 2)]].sum(axis=1)
df.drop(col_names, axis=1, inplace=True)


# col_names = [f'img_MRI_{i}' for i in range(1, 101)]
# df['Brain MRIs'] = df[col_names].sum(axis=1)
# df.drop(col_names, axis=1, inplace=True)

# import toml
# hmap = toml.load('./data/diagnostic_info.toml')['feature']
df_tmp = pd.read_csv('/home/skowshik/nacc_variable_vj.csv')
hmap = dict(zip(
    df_tmp.id, df_tmp.descriptor
))

# %% load testing data

from data.dataset_csv import CSVDataset
# basedir="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
basedir = '..'
fname = 'nacc_test_with_np_cli'
dat_file = f'{basedir}/data/train_vld_test_split_updated/{fname}.csv'
cnf_file = f'{basedir}/dev/data/toml_files/default_conf_new.toml'
img_net="SwinUNETREMB"
img_mode=1
mri_type="SEQ"
nacc_mri_info = "nacc_mri_3d.json"
other_mri_info = "other_3d_mris.json"
# emb_path = '/data_1/dlteif/SwinUNETR_MRI_emb/'
emb_path = '/data_1/skowshik/SwinUNETR_MRI_stripped_MNI_emb/'
print('Done.\nLoading testing dataset ...')

dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type,  emb_path=emb_path, specific_date=None, nacc_mri_info=nacc_mri_info, other_mri_info=other_mri_info, stripped='_stripped_MNI')

# %%

df_dat = pd.DataFrame([item[0] for item in dat_tst])

# remove MRI columns
col_names_sub = [n for n in df_dat.columns if n.startswith('img')]
df_dat = df_dat.drop(col_names_sub, axis=1)
df_dat['Brain MRI (T1)'] = [0] * len(df_dat)
df_dat['Brain MRI (T2)'] = [0] * len(df_dat)
df_dat['Brain MRI (FLAIR)'] = [0] * len(df_dat)
df_dat['Brain MRI (SWI)'] = [0] * len(df_dat)


# %%

from matplotlib import rc, rcParams
rc('axes', linewidth=0.5)
rc('font', size=15)
rcParams['font.family'] = 'Arial'
fontsize = 7

def get_order(df, label):
    # select label
    df_tmp = df[df.label == label]
    # df_tmp = df[(df.label == label) & df.ground_truth == 1]

    # mri only?
    # df_tmp = df_tmp[(df_tmp['Brain MRI (T1)'] != 0) | (df_tmp['Brain MRI (T2)'] != 0) | (df_tmp['Brain MRI (FLAIR)'] != 0) | (df_tmp['Brain MRI (SWI)'] != 0)]
    # print(df_tmp)

    # select true positives
    df_tmp = df_tmp[(df_tmp['logits'] > 0)]

    # select top n on logits
    # df_tmp = df_tmp.sort_values('logits', ascending=False)
    df_tmp['num_features'] = df_tmp.count(axis=1)
    df_tmp = df_tmp.sort_values(['num_features', 'logits'], ascending=False)
    df_tmp = df_tmp.drop(['num_features'], axis=1)

    th = df_tmp.logits.iloc[500 if len(df_tmp) > 500 else len(df_tmp) - 1]
    df_tmp = df_tmp[df_tmp.logits > max(th, 0)]

    # selected cases
    ids = df_tmp['index']
    # print(ids)

    #
    # print(df_tmp)
    # print(ids)
    df_tmp = df_tmp[df_tmp.columns[4:]]
    # print(f"# of cases to order: {len(df_tmp)}")

    # sort
    col_avg = df_tmp.mean()
    col_avg = col_avg.sort_values(ascending=False)

    df_tmp = df_tmp[col_avg.index]

    return col_avg, ids, df_tmp

def get_data(df, label, df_dat):
    df_tmp = df_dat.copy()

    # sort columns
    order, ids, _ = get_order(df, label)
    df_tmp = df_tmp[order.index]
    df_tmp = df_tmp.iloc[ids.to_numpy()]
    # print(ids)
    # print('df_dat\n', df_tmp)
    return df_tmp.to_numpy()


palette_label = {
    'NC': (30/255, 136/255, 229/255, 1.0),
    'MCI': (255/255, 193/255, 7/255, 1.0),
    'DE': (216/255, 27/255, 96/255, 1.0)
}

# for label in ['AD', 'LBD', 'VD', 'FTD', 'NPH', 'PSY', 'SEF', 'TBI', 'PRD', 'ODE']:
for label in ['NC', 'MCI', "DE"]:
# for label in ['NC']:
# for label in ['DE']:
    print(label)
    # sort features
    order, ids, df_tmp = get_order(df, label)

    # # replace 0 Shapley value
    # df_tmp = df_tmp.replace(0, np.nan)

    # remove if the feature list is long
    # n_features_to_keep = 10
    # df_tmp = df_tmp.drop(['NONFREQ'], axis=1)

    feature_names = []
    for fea in df_tmp.columns:
        tmp = fea.split('_')
        k = tmp[1] if len(tmp) > 1 else fea
        name = hmap[k] if k in hmap else k
        try:
            feature_names.append(name)
        except:
            feature_names.append(fea)
        # feature_names.append(fea)
    mock_shap_values = df_tmp.to_numpy()
    # clip shap values
    mock_shap_values[mock_shap_values < -.5] = -.5
    mock_shap_values[mock_shap_values > 1] = 1
    # mock_shap_values[mock_shap_values == 0] = 'nan'

    # feature values
    mock_feature_values = get_data(df, label, df_dat)
    mock_feature_values = mock_feature_values.astype(np.float32)
    # print(mock_feature_values.shape)

    base_values = np.zeros(mock_shap_values.shape[1])
    top_n = 20
    mock_explanation = shap.Explanation(
        values = mock_shap_values[:, :top_n],
        base_values = np.array([0]),
        data = mock_feature_values[:, :top_n],
        feature_names = feature_names[:top_n]
    )

    # convert data to df, and normalize
    df_foo = pd.DataFrame(
        data = mock_explanation.data,
        columns = mock_explanation.feature_names
    )
    df_foo = (df_foo - df_foo.min()) / (df_foo.max() - df_foo.min())

    # to address the plotting bugs in SHAP package
    # if don't do so, all dots appear gray for unknown reasons
    if 'Difficulty/need help assembling tax records, business affairs, or other paper ' in df_foo.columns:
        tmp = df_foo['Difficulty/need help assembling tax records, business affairs, or other paper '].to_numpy()
        tmp[:] = 0.5
        df_foo['Difficulty/need help assembling tax records, business affairs, or other paper '] = tmp

    if "Difficulty/need help writing checks, paying bills, or balancing a checkbook " in df_foo.columns:
        tmp = df_foo["Difficulty/need help writing checks, paying bills, or balancing a checkbook "].to_numpy()
        tmp[:] = 0.5
        df_foo["Difficulty/need help writing checks, paying bills, or balancing a checkbook "] = tmp

    mock_explanation.data = df_foo

    # stupid code, but I have to do
    # mock_explanation.data[pd.isna(mock_explanation.data)] = -0.1

    # custom colormap
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    cool_colormap = plt.get_cmap('cool')
    gray_colormap = ListedColormap(['gray'])
    combined_colors = np.vstack((gray_colormap(np.linspace(-0.01, 1, int(0.2 * 256))),
                                cool_colormap(np.linspace(-0.01, 1, int(0.8 * 256)))))
    combined_colormap = LinearSegmentedColormap.from_list('custom_cool_gray', combined_colors)

    # values_to_order = np.nanmean(np.abs(mock_shap_values), axis=0)
    # values_to_order[np.isnan(values_to_order)] = 0
    wellll = mock_explanation.mean(axis=0)
    wellll.values = np.arange(len(feature_names))[::-1]
    ax = shap.plots.beeswarm(
        mock_explanation,
        max_display = top_n,
        color_bar = False,
        show = False,
        s=4,
        plot_size=(8, 3),
        # order = list(range(len(feature_names)))[::-1]
        # color = palette_label[label] if label in palette_label else palette_label['NC']
        # order = mock_explanation.abs.mean(axis=0)
        order = wellll,
    )

    # change text
    # yticklabels = ax.get_yticklabels()
    # yticklabels[0] = 'Other features'
    # ax.set_yticklabels(yticklabels)

    ax.set_xlabel(f"Shapley value ({label})", fontsize=fontsize)

    pos = ax.get_position()
    new_pos = [0.02, 0.15, .3, .82]
    ax.set_position(new_pos)

    # wrap text
    yticklabels = ax.get_yticklabels()
    yticklabels = [textwrap.fill(label.get_text(),
        # width = 60 if len(label.get_text()) > 80 else 1000
        width = 120
    ) for label in yticklabels]
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_ticks_position('right')

    # font size
    for i, txt in enumerate(ax.get_yticklabels()):
        txt.set_fontsize(fontsize)
    for i, txt in enumerate(ax.get_xticklabels()):
        txt.set_fontsize(fontsize)

    # plt.clim(0, 1)
    ax.set_xlim([-0.52, 1.02])

    ax.set_xticks([-.5, 0, .5, 1])
    ax.set_xticklabels([r'$\leq$-0.5', '0', '0.5', r'$\geq$1'])

    cb = plt.colorbar(ax = [ax], location='left', values=None, aspect=50, extend='neither')
    ticks = cb.get_ticks()
    print(ticks)
    cb.set_ticks(ticks=[ticks[0],ticks[-1]], labels=['Low', 'High'], fontsize=fontsize)
    cb.ax.get_yaxis().labelpad = 5
    cb.ax.set_ylabel('Feature value', rotation=90, size=fontsize)

    # adjust size, to help with the text overlapping issue
    fig = plt.gcf()
    # fig.set_size_inches(16, 5 + 5)

    # plt.tight_layout()
    plt.savefig(f'../plots/final_figs/fig_shap_{label}.svg')
    plt.show()

    # if label == 'LBD':
    #     break





# %%
