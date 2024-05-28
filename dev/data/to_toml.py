#%%
features = ['his_NACCREAS', 'his_NACCREFR', 'his_BIRTHMO', 'his_BIRTHYR', 'his_HISPANIC', 'his_HISPOR', 'his_RACE', 'his_RACESEC', 'his_RACETER', 'his_PRIMLANG', 'his_EDUC', 'his_MARISTAT', 'his_LIVSIT', 'his_INDEPEND', 'his_RESIDENC', 'his_HANDED', 'his_NACCAGE', 'his_NACCNIHR', 'his_NACCFAM', 'his_NACCMOM', 'his_NACCDAD', 'his_NACCFADM', 'his_NACCAM', 'his_NACCAMS', 'his_NACCFFTD', 'his_NACCFM', 'his_NACCFMS', 'his_NACCOM', 'his_NACCOMS', 'his_TOBAC30', 'his_TOBAC100', 'his_SMOKYRS', 'his_PACKSPER', 'his_QUITSMOK', 'his_ALCOCCAS', 'his_ALCFREQ', 'his_CVHATT', 'his_HATTMULT', 'his_HATTYEAR', 'his_CVAFIB', 'his_CVANGIO', 'his_CVBYPASS', 'his_CVPACDEF', 'his_CVPACE', 'his_CVCHF', 'his_CVANGINA', 'his_CVHVALVE', 'his_CVOTHR', 'his_CBSTROKE', 'his_STROKMUL', 'his_NACCSTYR', 'his_CBTIA', 'his_TIAMULT', 'his_NACCTIYR', 'his_PD', 'his_PDYR', 'his_PDOTHR', 'his_PDOTHRYR', 'his_SEIZURES', 'his_TBI', 'his_TBIBRIEF', 'his_TRAUMBRF', 'his_TBIEXTEN', 'his_TRAUMEXT', 'his_TBIWOLOS', 'his_TRAUMCHR', 'his_TBIYEAR', 'his_NCOTHR', 'his_DIABETES', 'his_DIABTYPE', 'his_HYPERTEN', 'his_HYPERCHO', 'his_B12DEF', 'his_THYROID', 'his_ARTHRIT', 'his_ARTHTYPE', 'his_ARTHSPIN', 'his_ARTHUNK', 'his_INCONTU', 'his_INCONTF', 'his_APNEA', 'his_RBD', 'his_INSOMN', 'his_OTHSLEEP', 'his_ALCOHOL', 'his_ABUSOTHR', 'his_PTSD', 'his_BIPOLAR', 'his_SCHIZ', 'his_DEP2YRS', 'his_DEPOTHR', 'his_ANXIETY', 'his_OCD', 'his_NPSYDEV', 'his_PSYCDIS', 'his_NACCTBI', 'med_ANYMEDS', 'med_NACCAMD', 'med_NACCAHTN', 'med_NACCHTNC', 'med_NACCACEI', 'med_NACCAAAS', 'med_NACCBETA', 'med_NACCCCBS', 'med_NACCDIUR', 'med_NACCVASD', 'med_NACCANGI', 'med_NACCLIPL', 'med_NACCNSD', 'med_NACCAC', 'med_NACCADEP', 'med_NACCAPSY', 'med_NACCADMD', 'med_NACCPDMD', 'med_NACCEMD', 'med_NACCEPMD', 'med_NACCDBMD', 'ph_HEIGHT', 'ph_WEIGHT', 'ph_NACCBMI', 'ph_BPSYS', 'ph_BPDIAS', 'ph_HRATE', 'ph_VISION', 'ph_VISCORR', 'ph_VISWCORR', 'ph_HEARING', 'ph_HEARAID', 'ph_HEARWAID', 'cvd_ABRUPT', 'cvd_STEPWISE', 'cvd_SOMATIC', 'cvd_EMOT', 'cvd_HXHYPER', 'cvd_HXSTROKE', 'cvd_FOCLSYM', 'cvd_FOCLSIGN', 'cvd_HACHIN', 'cvd_CVDCOG', 'cvd_STROKCOG', 'cvd_CVDIMAG', 'cvd_CVDIMAG1', 'cvd_CVDIMAG2', 'cvd_CVDIMAG3', 'cvd_CVDIMAG4', 'updrs_PDNORMAL', 'updrs_SPEECH', 'updrs_FACEXP', 'updrs_TRESTFAC', 'updrs_TRESTRHD', 'updrs_TRESTLHD', 'updrs_TRESTRFT', 'updrs_TRESTLFT', 'updrs_TRACTRHD', 'updrs_TRACTLHD', 'updrs_RIGDNECK', 'updrs_RIGDUPRT', 'updrs_RIGDUPLF', 'updrs_RIGDLORT', 'updrs_RIGDLOLF', 'updrs_TAPSRT', 'updrs_TAPSLF', 'updrs_HANDMOVR', 'updrs_HANDMOVL', 'updrs_HANDALTR', 'updrs_HANDALTL', 'updrs_LEGRT', 'updrs_LEGLF', 'updrs_ARISING', 'updrs_POSTURE', 'updrs_GAIT', 'updrs_POSSTAB', 'updrs_BRADYKIN', 'npiq_NPIQINF', 'npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP', 'gds_NOGDS', 'gds_SATIS', 'gds_DROPACT', 'gds_EMPTY', 'gds_BORED', 'gds_SPIRITS', 'gds_AFRAID', 'gds_HAPPY', 'gds_HELPLESS', 'gds_STAYHOME', 'gds_MEMPROB', 'gds_WONDRFUL', 'gds_WRTHLESS', 'gds_ENERGY', 'gds_HOPELESS', 'gds_BETTER', 'gds_NACCGDS', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE', 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL', 'exam_NORMEXAM', 'exam_FOCLDEF', 'exam_GAITDIS', 'exam_EYEMOVE', 'exam_PARKSIGN', 'exam_RESTTRL', 'exam_RESTTRR', 'exam_SLOWINGL', 'exam_SLOWINGR', 'exam_RIGIDL', 'exam_RIGIDR', 'exam_BRADY', 'exam_PARKGAIT', 'exam_POSTINST', 'exam_CVDSIGNS', 'exam_CORTDEF', 'exam_SIVDFIND', 'exam_CVDMOTL', 'exam_CVDMOTR', 'exam_CORTVISL', 'exam_CORTVISR', 'exam_SOMATL', 'exam_SOMATR', 'exam_POSTCORT', 'exam_PSPCBS', 'exam_EYEPSP', 'exam_DYSPSP', 'exam_AXIALPSP', 'exam_GAITPSP', 'exam_APRAXSP', 'exam_APRAXL', 'exam_APRAXR', 'exam_CORTSENL', 'exam_CORTSENR', 'exam_ATAXL', 'exam_ATAXR', 'exam_ALIENLML', 'exam_ALIENLMR', 'exam_DYSTONL', 'exam_DYSTONR', 'exam_MYOCLLT', 'exam_MYOCLRT', 'exam_ALSFIND', 'exam_GAITNPH', 'exam_OTHNEUR', 'bat_MMSECOMP', 'bat_MMSELOC', 'bat_MMSELAN', 'bat_MMSEVIS', 'bat_MMSEHEAR', 'bat_MMSEORDA', 'bat_MMSEORLO', 'bat_PENTAGON', 'bat_NACCMMSE', 'bat_NPSYCLOC', 'bat_NPSYLAN', 'bat_LOGIMO', 'bat_LOGIDAY', 'bat_LOGIYR', 'bat_LOGIPREV', 'bat_LOGIMEM', 'bat_MEMUNITS', 'bat_MEMTIME', 'bat_UDSBENTC', 'bat_UDSBENTD', 'bat_UDSBENRS', 'bat_DIGIF', 'bat_DIGIFLEN', 'bat_DIGIB', 'bat_DIGIBLEN', 'bat_ANIMALS', 'bat_VEG', 'bat_TRAILA', 'bat_TRAILARR', 'bat_TRAILALI', 'bat_TRAILB', 'bat_TRAILBRR', 'bat_TRAILBLI', 'bat_WAIS', 'bat_BOSTON', 'bat_UDSVERFC', 'bat_UDSVERFN', 'bat_UDSVERNF', 'bat_UDSVERLC', 'bat_UDSVERLR', 'bat_UDSVERLN', 'bat_UDSVERTN', 'bat_UDSVERTE', 'bat_UDSVERTI', 'bat_COGSTAT', 'bat_MODCOMM', 'bat_MOCACOMP', 'bat_MOCAREAS', 'bat_MOCALOC', 'bat_MOCALAN', 'bat_MOCAVIS', 'bat_MOCAHEAR', 'bat_MOCATOTS', 'bat_NACCMOCA', 'bat_MOCATRAI', 'bat_MOCACUBE', 'bat_MOCACLOC', 'bat_MOCACLON', 'bat_MOCACLOH', 'bat_MOCANAMI', 'bat_MOCAREGI', 'bat_MOCADIGI', 'bat_MOCALETT', 'bat_MOCASER7', 'bat_MOCAREPE', 'bat_MOCAFLUE', 'bat_MOCAABST', 'bat_MOCARECN', 'bat_MOCARECC', 'bat_MOCARECR', 'bat_MOCAORDT', 'bat_MOCAORMO', 'bat_MOCAORYR', 'bat_MOCAORDY', 'bat_MOCAORPL', 'bat_MOCAORCT', 'bat_CRAFTVRS', 'bat_CRAFTURS', 'bat_DIGFORCT', 'bat_DIGFORSL', 'bat_DIGBACCT', 'bat_DIGBACLS', 'bat_CRAFTDVR', 'bat_CRAFTDRE', 'bat_CRAFTDTI', 'bat_CRAFTCUE', 'bat_MINTTOTS', 'bat_MINTTOTW', 'bat_MINTSCNG', 'bat_MINTSCNC', 'bat_MINTPCNG', 'bat_MINTPCNC', 'bat_MOCBTOTS', 'bat_NACCMOCB', 'bat_REY1REC', 'bat_REY1INT', 'bat_REY2REC', 'bat_REY2INT', 'bat_REY3REC', 'bat_REY3INT', 'bat_REY4REC', 'bat_REY4INT', 'bat_REY5REC', 'bat_REY5INT', 'bat_REY6REC', 'bat_REY6INT', 'bat_OTRAILA', 'bat_OTRLARR', 'bat_OTRLALI', 'bat_OTRAILB', 'bat_OTRLBRR', 'bat_OTRLBLI', 'bat_REYDREC', 'bat_REYDINT', 'bat_REYTCOR', 'bat_REYFPOS', 'bat_VNTTOTW', 'bat_VNTPCNC', 'bat_RESPVAL', 'bat_RESPHEAR', 'bat_RESPDIST', 'bat_RESPINTR', 'bat_RESPDISN', 'bat_RESPFATG', 'bat_RESPEMOT', 'bat_RESPASST', 'bat_RESPOTH', 'apoe_NACCNE4S', 'his_SEX', 'npiq_ANX', 'his_ARTHUPEX', 'his_ARTHLOEX', 'med_NACCAANX', 'exam_NACCNREX']
len(features)
#%%


labels = [
    'NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE'

]


import pandas as pd

path = 'meta_files/meta_file.csv'
df = pd.read_csv(path)
# output features
print('[features]')
print()
for i in range(len(df)):
    row = df.iloc[i]
    name = row['name'].strip()
    if name in features:
        type = row['type'].strip()
        print('\t[feature.{}]'.format(name))
        # if name == 'img_MRI_T1':
        #     print('\ttype = \"undefined\"')
        if type == 'C':
            print('\ttype = \"categorical\"')
            print('\tnum_categories = {}'.format(int(row['num_unique_values'])))
        elif type == 'N':
            print('\ttype = \"numerical\"')
            try:
                print('\tshape = [{}]'.format(int(row['length'])))
            except:
                print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
        elif type == 'M':
            print('\ttype = \"imaging\"')
            try:
                print('\tshape = [{}]'.format(int(row['length'])))
            except:
                print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
        print()

# output labels
print('[labels]')
print()
for name in labels:
    print('\t[label.{}]'.format(name))
    print('\ttype = \"categorical\"')
    print('\tnum_categories = 2')
    print()