
# %%
import pandas as pd
import numpy as np
import os
import re
from icecream import ic
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore")

# Train set
nacc = pd.read_csv('training_cohorts/new_nacc_revised_selection.csv')
aibl = pd.read_csv('training_cohorts/aibl_revised.csv')
nifd = pd.read_csv('training_cohorts/nifd_revised.csv')
ppmi = pd.read_csv('training_cohorts/ppmi_revised.csv')
stanford = pd.read_csv('training_cohorts/stanford_revised.csv')
oasis = pd.read_csv('training_cohorts/oasis_revised.csv')
rtni = pd.read_csv('training_cohorts/rtni_revised.csv')

# Test set
adni_df = pd.read_csv('train_vld_test_split_updated/adni_merged.csv')
fhs = pd.read_csv('train_vld_test_split_updated/fhs_converted_6months_cleaned.csv')
label_features = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']


# %%
def get_label_distribution(df, cohort):
    avail_lab = []
    for label in label_features:
        if label in df.columns:
            cnt_dict = dict(df[label].value_counts())
            if 1 in cnt_dict:
                label_dist[cohort][label] = cnt_dict[1]
                avail_lab.append(label)
                if label in label_dist['total']:
                    label_dist['total'][label] += cnt_dict[1]
                else:
                    label_dist['total'][label] = cnt_dict[1]
            else:
                label_dist[cohort][label] = 0
    
    label_dist[cohort]['total'] = len(df[(df[avail_lab] == 1) | (df[avail_lab] == 0)].dropna(how='all', subset=avail_lab))


label_dist = {'nacc': {}, 'aibl': {}, 'nifd': {}, 'ppmi': {}, 'stanford': {}, 'oasis': {}, 'rtni': {}, 'adni': {}, 'fhs': {}, 'total': {}}
get_label_distribution(nacc, 'nacc')
get_label_distribution(aibl, 'aibl')
get_label_distribution(nifd, 'nifd')
get_label_distribution(ppmi, 'ppmi')
get_label_distribution(stanford, 'stanford')
get_label_distribution(oasis, 'oasis')
get_label_distribution(rtni, 'rtni')
get_label_distribution(adni_df, 'adni')
get_label_distribution(fhs, 'fhs')
label_dist['total']['total'] = sum([label_dist[val]['total'] for val in label_dist.keys() if val != 'total'])
label_dist_df = pd.DataFrame(label_dist)
label_dist_df.to_csv('label_dist.csv')

# %%
def anova_test(df, col):
    avail_labels = [label for label in label_features if label in df.columns and len(df[df[label] == 1]) != 0 and label != 'DE']
    samples = [df[(~df[col].isna()) & (df[label] == 1)][col] for label in avail_labels]
    samples = [samp for samp in samples if len(samp) != 0]
    test_stat, p_value = stats.f_oneway(*samples)

    # if p_value < 0.001: return '<0.001'
    return '{:.3e}'.format(p_value)

def chi_square_test(df, col):
    avail_labels = [label for label in label_features if label in df.columns and len(df[df[label] == 1]) != 0 and label != 'DE']
    if col not in df.columns:
        return 'N.A'
    pool = []
    for lab in avail_labels:
        if lab == 'DE':
            continue
        pool.append(list(df[df[lab] == 1][col]))

    element = set([])
    for p in pool:
        for a in p:
            if isinstance(a, str) or (not math.isnan(a)):
                element.add(a)
    # print(element)
    matrix = []
    for p in pool:
        matrix.append([p.count(e) for e in element])
    # print(col, matrix)
    matrix = [row for row in matrix if any(item != 0 for item in row)]
    if not any(matrix):
        return 'N.A'
    chi2, p, dof, exp = stats.chi2_contingency(np.array(matrix))
    # if p < 0.001: return '<0.001'
    return '{:.3e}'.format(p)

def get_mean_std(df, col):
    mean = np.nanmean(df[col])
    std_dev = np.nanstd(df[col])
    # p_value + get_p_value(df, col, mean)
    return mean, std_dev #, p_value

def get_all_p_values(df, data=None):
    # print(data)
    p_dict = {'n': np.NaN}

    for col in ['his_NACCAGE', 'his_EDUC', 'bat_NACCMMSE', 'cdr_CDRGLOB', 'bat_NACCMOCA']:
        if col in df.columns:
            p_dict[col] = anova_test(df, col)
        else:
            p_dict[col] = 'N.A.'
    
    for col in ['his_SEX', 'his_NACCNIHR', 'his_PRIMLANG', 'his_MARISTAT', 'his_LIVSIT', 'apoe_NACCNE4S']:
        if col in df.columns:
            p_dict[col] = chi_square_test(df, col)
        else:
                p_dict[col] = 'N.A.'
        # print(data, col, p_dict[col])
    return p_dict

def get_statistics(df, data=None):
    stat_dict = {}
    for label in label_features:
        if label not in df.columns or label == 'DE':
            continue
        # print(f'{label}.....')
        df_lb = df[df[label] == 1]
        if len(df_lb) == 0:
            continue
        stat_dict[label] = {'n': len(df_lb)}

        for col in ['his_NACCAGE', 'his_EDUC', 'bat_NACCMMSE', 'cdr_CDRGLOB', 'bat_NACCMOCA']:
            if col in df.columns:
                if len(df_lb[~df_lb[col].isna()]) == 0:
                    stat_dict[label][col] = 'N.A.'
                else:
                    mean, std = get_mean_std(df_lb, col)
                    stat_dict[label][col] = f'{round(mean, 2)} +- {round(std, 2)}'
                    if len(df_lb[df_lb[col].isna()]) > 0:
                        stat_dict[label][col] += '^'
            else:
                stat_dict[label][col] = 'N.A.'


        if 'his_SEX' in df_lb.columns:
            if 'male' in df_lb['his_SEX'].value_counts():
                gen_male_cnt, gen_male_per = df_lb['his_SEX'].value_counts()['male'], round(df_lb['his_SEX'].value_counts(normalize=True)['male'] * 100, 2)
                stat_dict[label]['his_SEX'] = f'{gen_male_cnt}, {gen_male_per}%'
                if len(df_lb[df_lb['his_SEX'].isna()]) > 0:
                    stat_dict[label]['his_SEX'] += '^'
            else:
                stat_dict[label]['his_SEX'] = '0, 0%'
        else:
            stat_dict[label]['his_SEX'] = 'N.A'
            
        if 'his_NACCNIHR' in df_lb.columns:
            race_counts = dict(df_lb['his_NACCNIHR'].value_counts())
            whi = str(race_counts['whi']) if 'whi' in race_counts else '0'
            blk = str(race_counts['blk']) if 'blk' in race_counts else '0'
            asi = str(race_counts['asi']) if 'asi' in race_counts else '0'
            ind = str(race_counts['ind']) if 'ind' in race_counts else '0'
            haw = str(race_counts['haw']) if 'haw' in race_counts else '0'
            mul = str(race_counts['mul']) if 'mul' in race_counts else '0'
            stat_dict[label]['his_NACCNIHR'] = f"({whi}, {blk}, {asi}, {ind}, {haw}, {mul})"
            if len(df_lb[df_lb['his_NACCNIHR'].isna()]) > 0:
                stat_dict[label]['his_NACCNIHR'] += '^'
        else:
            stat_dict[label]['his_NACCNIHR'] = 'N.A.'
        
        if 'his_PRIMLANG' in df_lb.columns:
            race_counts = dict(df_lb['his_PRIMLANG'].value_counts())
            stat_dict[label]['his_PRIMLANG'] = str(race_counts)
            if len(df_lb[df_lb['his_PRIMLANG'].isna()]) > 0:
                stat_dict[label]['his_PRIMLANG'] += '^'
        else:
            stat_dict[label]['his_PRIMLANG'] = 'N.A.'
            
        if 'his_MARISTAT' in df_lb.columns:
            race_counts = dict(df_lb['his_MARISTAT'].value_counts())
            stat_dict[label]['his_MARISTAT'] = str(race_counts)
            if len(df_lb[df_lb['his_MARISTAT'].isna()]) > 0:
                stat_dict[label]['his_MARISTAT'] += '^'
        else:
            stat_dict[label]['his_MARISTAT'] = 'N.A.'
            
        if 'his_LIVSIT' in df_lb.columns:
            race_counts = dict(df_lb['his_LIVSIT'].value_counts())
            stat_dict[label]['his_LIVSIT'] = str(race_counts)
            if len(df_lb[df_lb['his_LIVSIT'].isna()]) > 0:
                stat_dict[label]['his_LIVSIT'] += '^'
        else:
            stat_dict[label]['his_LIVSIT'] = 'N.A.'

        if 'apoe_NACCNE4S' in df_lb.columns:
            apoe = df_lb['apoe_NACCNE4S'].value_counts()
            apoe_1 = apoe[1.0] if 1.0 in apoe else 0
            apoe_2 = apoe[2.0] if 2.0 in apoe else 0
            apoe_cnt = apoe_1 + apoe_2
            apoe_per = round((apoe_cnt / sum(df_lb['apoe_NACCNE4S'].value_counts())) * 100, 2)
            stat_dict[label]['apoe_NACCNE4S'] = f'{apoe_cnt}, {apoe_per}%'
            if len(df_lb[df_lb['apoe_NACCNE4S'].isna()]) > 0:
                stat_dict[label]['apoe_NACCNE4S'] += '^'
        else:
            stat_dict[label]['apoe_NACCNE4S'] = 'N.A.'


        stat_dict[label] = {'n': stat_dict[label]['n'], 'his_NACCAGE': stat_dict[label]['his_NACCAGE'], 'his_SEX': stat_dict[label]['his_SEX'], 'his_EDUC': stat_dict[label]['his_EDUC'], 'his_NACCNIHR': stat_dict[label]['his_NACCNIHR'], 'cdr_CDRGLOB': stat_dict[label]['cdr_CDRGLOB'], 'bat_NACCMMSE': stat_dict[label]['bat_NACCMMSE'], 'bat_NACCMOCA': stat_dict[label]['bat_NACCMOCA'], 'apoe_NACCNE4S': stat_dict[label]['apoe_NACCNE4S'], 'his_PRIMLANG': stat_dict[label]['his_PRIMLANG'], 'his_MARISTAT': stat_dict[label]['his_MARISTAT'], 'his_LIVSIT': stat_dict[label]['his_LIVSIT']}

    stat_dict['p_value'] = get_all_p_values(df, data)
    # print(stat_dict['p_value'])
    return stat_dict
        

# %%
nacc_stat_dict = get_statistics(nacc)
nacc_stat_df = pd.DataFrame(nacc_stat_dict).T
nacc_stat_df['cohort'] = 'NACC'

nifd_stat_dict = get_statistics(nifd)
nifd_stat_df = pd.DataFrame(nifd_stat_dict).T
nifd_stat_df['cohort'] = 'NIFD'

ppmi_stat_dict = get_statistics(ppmi)
ppmi_stat_df = pd.DataFrame(ppmi_stat_dict).T
ppmi_stat_df['cohort'] = 'PPMI'

aibl_stat_dict = get_statistics(aibl)
aibl_stat_df = pd.DataFrame(aibl_stat_dict).T
aibl_stat_df['cohort'] = 'AIBL'

oasis_stat_dict = get_statistics(oasis)
oasis_stat_df = pd.DataFrame(oasis_stat_dict).T
oasis_stat_df['cohort'] = 'OASIS'

stanford_stat_dict = get_statistics(stanford)
stanford_stat_df = pd.DataFrame(stanford_stat_dict).T
stanford_stat_df['cohort'] = 'LBDSU'

rtni_stat_dict = get_statistics(rtni)
rtni_stat_df = pd.DataFrame(rtni_stat_dict).T
rtni_stat_df['cohort'] = '4RTNI'

adni_stat_dict = get_statistics(adni_df)
adni_stat_df = pd.DataFrame(adni_stat_dict).T
adni_stat_df['cohort'] = 'ADNI'

fhs_stat_dict = get_statistics(fhs)
fhs_stat_df = pd.DataFrame(fhs_stat_dict).T
fhs_stat_df['cohort'] = 'FHS'

combined_stat_df = pd.concat([nacc_stat_df, nifd_stat_df, ppmi_stat_df, aibl_stat_df, oasis_stat_df, stanford_stat_df, rtni_stat_df, adni_stat_df, fhs_stat_df], axis=0)
combined_stat_df = combined_stat_df[['cohort'] + list(combined_stat_df.columns)[:-1]]
combined_stat_df.to_csv('stats.csv', index=True)

# %%
print(combined_stat_df)


# %%
train = pd.read_csv('train_vld_test_split_updated/nacc_train.csv')
vld = pd.read_csv('train_vld_test_split_updated/nacc_vld.csv')
test = pd.read_csv('train_vld_test_split_updated/nacc_test_with_np_cli.csv')
merged_train = pd.read_csv('train_vld_test_split_updated/merged_train.csv')
merged_vld = pd.read_csv('train_vld_test_split_updated/merged_vld.csv')
merged = pd.read_csv('training_cohorts/merged_data_nacc_nifd_stanford_aibl_ppmi_oasis_rtni.csv')
nacc_np = pd.read_csv('train_vld_test_split_updated/nacc_neuropath.csv')
nacc_np = nacc[nacc['ID'].isin(nacc_np['ID'])]
cli = pd.read_csv('train_vld_test_split_updated/clinician_review_cases_test.csv')
cli = nacc[nacc['ID'].isin(cli['ID'])]

merged_ = pd.concat([merged, cli, nacc_np], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
merged_.drop_duplicates(inplace=True)
# cli = pd.read_csv('testing_cohorts/clinician_review_cases_test.csv')
# radio = pd.read_csv('testing_cohorts/radiologist_review_cases_test.csv')
# neuropath = pd.read_csv('testing_cohorts/neuropath_cases_with_np_fea.csv')
print(len(merged_))
print(len(merged_train) + len(merged_vld) + len(test))
labels_dict = {'nacc_dict': {}, 'nacc_trn_dict': {}, 'nacc_vld_dict': {}, 'nacc_tst_dict': {}, 'merged_dict': {}, 'merged_trn_dict': {}, 'merged_vld_dict': {}, 'merged_tst_dict': {}}
for label in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
    labels_dict['nacc_dict'][label] = dict(nacc[label].value_counts())
    labels_dict['nacc_trn_dict'][label] = dict(train[label].value_counts())
    labels_dict['nacc_vld_dict'][label] = dict(vld[label].value_counts())
    labels_dict['nacc_tst_dict'][label] = dict(test[label].value_counts())
    labels_dict['merged_dict'][label] = dict(merged_[label].value_counts())
    labels_dict['merged_trn_dict'][label] = dict(merged_train[label].value_counts())
    labels_dict['merged_vld_dict'][label] = dict(merged_vld[label].value_counts())
    labels_dict['merged_tst_dict'][label] = dict(test[label].value_counts())
    
labels_split_df = pd.DataFrame(labels_dict)
labels_split_df.to_csv('labels_split_dist.csv')
print(labels_split_df)
# %%
