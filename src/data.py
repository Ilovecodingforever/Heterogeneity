import os
import sys
import copy
import itertools

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy

from sklearn import tree

import auton_survival
from auton_survival.reporting import plot_kaplanmeier
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.metrics import phenotype_purity
from auton_survival.models.dcm import DeepCoxMixtures
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import brier_score
from auton_survival.models.dcm.dcm_utilities import predict_latent_z
from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects

from model import phenotyping



def get_characteristics(data, phenotype, cat_feats, num_feats):
    pd.set_option('display.max_rows', 500)
    assert(len(np.unique(phenotype) == 2))
    
    num_characteristics = data[num_feats].groupby(phenotype).describe().T
    mean_std = num_characteristics.loc[(slice(None), ['mean', 'std']), :]
    ttest = data[num_feats].apply(lambda x: scipy.stats.ttest_ind(x[phenotype == 0].dropna(), x[phenotype == 1].dropna(), equal_var=False).pvalue, axis=0)
    print('mean and std for numerical features:')
    print(mean_std)
    print('t-test for numerical features, p-values:')
    print(ttest)
    
    lst = [pd.crosstab(data[feat], phenotype) for feat in cat_feats]
    # same:
    # lst = [(data[feat].groupby(phenotype).value_counts()).unstack(level=0) for feat in cat_feats]
    counts = pd.concat(lst, keys=cat_feats, names=['feature', 'value'])
    chisq = counts.groupby('feature').apply(lambda x: scipy.stats.chi2_contingency(x).pvalue)
    print('counts for categorical features:')
    print(counts)
    print('chi-squared test for categorical features, p-values:')
    print(chisq)
    
    return




def bari2d():
    # https://www.sciencedirect.com/science/article/pii/S0735109713007961
    # Angina worked?

    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'BARI2D/data'

    data_file = 'bari2d_bl.sas7bdat'
    outcome_file = 'bari2d_endpts.sas7bdat'
    # outcome_file = 'bari2d_long.sas7bdat'

    data_path = os.path.join(directory, data_dir, data_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)

    dataset_raw = pd.read_sas(data_path).set_index('id')
    outcome = pd.read_sas(outcome_path).set_index('id')

    intervention = dataset_raw['cardtrt'].rename('intervention')
    # 0=Medical Therapy, 1=Early Revascularization
    # intervention.loc[intervention['intervention'] == 0, 'intervention'] = 'Medical Therapy (control)'
    # intervention.loc[intervention['intervention'] == 1, 'intervention'] = 'Early Revascularization (treatment)'

    # sex = dataset_raw[['sex']]
    # 1 male, 2 female
    # sex.loc[sex['sex'] == 2, 'sex'] = 'Female'
    # sex.loc[sex['sex'] == 1, 'sex'] = 'Male'

    e ='death'
    t = 'deathfu'
    outcome = outcome[[e, t]].rename(columns={e:'event', t:'time'})
    outcome['time'] /= 365.25
    
    # Angina:
    # outcome = outcome[['dayvisit', 'angcls']].rename(columns={'dayvisit':'time', 'angcls':'event'})
    # outcome['event'][outcome['event'] != 3] = 0
    # outcome['event'][outcome['event'] == 3] = 1
    # outcome['time'] /= 365.25

    # o_ = outcome.set_index('time', append=True)
    # # first time that the patient have outcome
    # idx = o_['event'] == 1
    # have_outcome = o_.loc[idx].reset_index().groupby('id').min('time')

    # # last time that patient not have outcome
    # idx = o_['event'] == 0
    # no_outcome = o_.loc[idx].reset_index().groupby('id').max('time')

    # # if patient experienced the outcome, set it to have_outcome. if not keep it no_outcome
    # outcome_ = no_outcome.copy()
    # outcome_.loc[have_outcome.index] = have_outcome
    # outcome_ = outcome_.reset_index().set_index('id')

    cat_feats = ['sex',
                'race', 'hxmi', 'hxhtn',
                # 'angcls6w', 'smkcat', 'ablvef', 'geographic_reg'
                ]
    num_feats = ['age', 'bmi', 
                #  'dmdur'
                 ]

    phenotypes, model = phenotyping(outcome, dataset_raw.loc[outcome.index][cat_feats + num_feats], intervention.loc[outcome.index], 
                        cat_feats, num_feats, name='Bari2D')

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) <= 6]
    num_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) > 6]
    print('Bari2D characteristics:')
    get_characteristics(dataset_raw.loc[outcome.index], phenotypes, cat_feats, num_feats)

    return phenotypes, model



def sts(model):
    data_raw = pd.read_csv('src/STS_preprocessing_files/timetoevent_cabg.csv')
    data = data_raw.copy()

    outcome_names = [
        # 'CTB', # ceased to breathe
                     'ReadmitFlag', 'ReadmitDays',
                     'NonCardiacFlag', 'NonCardiacReadmitDays',
                     'CardiacFlag', 'CardiacReadmitDays',
                     'MIFlag', 'MIReadmitDays',
                     'ACSFlag', 'ACSReadmitDays',
                     'AFFlag', 'AFReadmitDays',
                     'HFFlag', 'HFReadmitDays',
                     'StrokeFlag', 'StrokeReadmitDays',
                     'H_StrokeFlag', 'H_StrokeReadmitDays',
                     'I_StrokeFlag', 'I_StrokeReadmitDays',
                     'TIAFlag', 'TIAReadmitDays',
                     'MACCEFlag', 'MACCEReadmitDays',
                     'DrugFlag', 'DrugReadmitDays',
                     'EndoFlag', 'ENDOReadmitDays',
                     'ARRESTFlag', 'ARRESTReadmitDays',
                     'VTACHFlag', 'VTACHReadmitDays',
                     'VFIBFlag', 'VFIBReadmitDays',
                     'PCI_Flag', 'PCI_Days',
                     'CABG_Flag', 'CABG_Days',
                     'VAD_Flag', 'VAD_Days',
                     'Heart_Transplant_Flag', 'Heart_Transplant_Days']

    data['CTBFlag'] = data['CTB']
    data['CTBDays'] = data[[s for s in outcome_names if 'Days' in s]].max(axis=1)
    data.drop('CTB', axis=1, inplace=True)
    outcome_names = ['CTBFlag', 'CTBDays'] + outcome_names

    # filter patients with type 2 diabetes mellitus (T2DM) and stable coronary artery disease (CAD)
    data = data[(data['diabetes'] == 1) & (data['status'] == 1)]

    outcome = data[['ID'] + outcome_names].set_index('ID')

    # get an outcome
    outcome_idx = 0
    outcome = outcome[[outcome_names[outcome_idx*2], outcome_names[outcome_idx*2+1]]].rename(
        columns={outcome_names[outcome_idx*2]: 'event',
                 outcome_names[outcome_idx*2+1]: 'time'})
    outcome = outcome.dropna()
    outcome['time'] /= 365.25

    dataset = data.drop(outcome_names, axis=1).set_index('ID').loc[outcome.index]

    dataset['female'] += 1      # Female: 2, Male: 1
    dataset['prevmi'] = np.clip(dataset['prevmi'], 0, 1)
    dataset['hypertn'] = np.clip(dataset['hypertn'], 0, 1)
    

    cat_feats = ['female', 'racecaucasian', 'prevmi', 'hypertn', 
                #  'CardSympTimeOfAdm', 'Recentishsmoker', 
                # 'ablvef', 'geographic_reg'
                ]
    num_feats = ['age', 'bmi',
                #  'dmdur'
                ]

    phenotypes, model = phenotyping(outcome, dataset[cat_feats+num_feats], None, cat_feats, num_feats, model, name='STS')

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset.columns if len(dataset[c].unique()) <= 6]
    num_feats = [c for c in dataset.columns if len(dataset[c].unique()) > 6]
    print('STS characteristics:')
    get_characteristics(dataset, phenotypes, cat_feats, num_feats)

    return phenotypes, model


def accord():
    # https://www.sciencedirect.com/science/article/pii/S2405500X20307167
    # https://diabetesjournals.org/care/article/34/Supplement_2/S107/27581/The-ACCORD-Action-to-Control-Cardiovascular-Risk
    # https://www.nejm.org/doi/full/10.1056/nejmoa1001282
    # Lipid Placebo vs Lipid Fibrate should work for censor_tm, but not significant

    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'ACCORD/ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets'

    gender_file = 'accord_key.sas7bdat'
    outcome_file = 'cvdoutcomes.sas7bdat'

    event_var = 'censor_tm'
    time_var = 'fuyrs_tm'

    data_path = os.path.join(directory, data_dir, gender_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)

    dataset = pd.read_sas(data_path)
    outcome = pd.read_sas(outcome_path)

    treatments = [
        # [b'Standard Gylcemia/Intensive BP',
        # b'Standard Gylcemia/Standard BP'],

        # [b'Intensive Gylcemia/Intensive BP',
        # b'Intensive Gylcemia/Standard BP',],

       [b'Intensive Glycemia/Lipid Placebo',
        b'Intensive Glycemia/Lipid Fibrate',

        b'Standard Glycemia/Lipid Placebo',
        b'Standard Glycemia/Lipid Fibrate',]
    ]

    assert(len(outcome['MaskID'].unique()) == len(outcome))
    assert(len(dataset['MaskID'].unique()) == len(dataset))

    # threshold_yr = 5
    # outcome = outcome[outcome['fuyrs_po'] <= threshold_yr]
    sex = dataset.set_index('MaskID')[['female']].rename(columns={'female':'sex'})
    # 1 is female, 0 is male
    sex['sex'][sex['sex'] == 1] = 'Female'
    sex['sex'][sex['sex'] == 0] = 'Male'

    outcome = outcome.set_index('MaskID')[[event_var, time_var]].rename(columns={event_var: 'event',
                                                                                 time_var: 'time'})

    treatment = dataset.set_index('MaskID')[['treatment']].rename(columns={'treatment': 'intervention'})

    treatment = treatment[treatment['intervention'].isin(treatments[0])]

    dataset = sex.join(outcome, how='inner').join(treatment, how='inner')
    dataset['intervention'][(dataset['intervention'] == b'Intensive Glycemia/Lipid Placebo') | (dataset['intervention'] == b'Standard Glycemia/Lipid Placebo')] = 'Placebo'
    dataset['intervention'][(dataset['intervention'] == b'Intensive Glycemia/Lipid Fibrate') | (dataset['intervention'] == b'Standard Glycemia/Lipid Fibrate')] = 'Fibrate'

    dataset['condition'] = [", ".join([str(s), str(i)]) for s, i in dataset[['sex', 'intervention']].values]

    return dataset, 'condition'



def allhat():
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4114223/
    # The only significant treatment gender interaction noted was for cancer mortality for amlodipine versus chlorthalidone.
    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'ALLHAT/ALLHAT_v2016a/DATA/Forms'

    gender_file = 'ah01.sas7bdat'
    outcome_file = 'ah40.sas7bdat' # 'ah23.sas7bdat'
    time_file = 'ah07.sas7bdat'
    treatment_file = 'ah25.sas7bdat'

    data_path = os.path.join(directory, data_dir, gender_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)
    time_path = os.path.join(directory, data_dir, time_file)
    treatment_path = os.path.join(directory, data_dir, treatment_file)



    dataset = pd.read_sas(data_path).set_index('allhatid')
    outcome = pd.read_sas(outcome_path).set_index('allhatid')
    time = pd.read_sas(time_path).set_index('allhatid')
    treatment = pd.read_sas(treatment_path).set_index('allhatid')


    time = pd.read_sas(os.path.join(directory, data_dir, 'ah22.sas7bdat')).set_index('allhatid')[['F22FD024']]

    # F25FD027: CCB, F25FD023: Diuretics
    treatment['intervention'] = np.nan
    treatment['intervention'][(treatment['F25FD027'] == 1)] = 'CCB'
    treatment['intervention'][(treatment['F25FD023'] == 1)] = 'Diuretics'
    treatment['intervention'][(treatment['F25FD027'] == 1) & (treatment['F25FD023'] == 1)] = np.nan
    treatment = treatment['intervention'].dropna()

    # 'F23FD023'  Event type - hospitalized fatal
    # F23FD022, Num 8 Event date(mm/dd/yy)
    # F01FD076, 2 female
    # ah01, F01FD037, Visit 1 date (mm-dd-yy)

    # female is 2, male is 1
    sex = dataset[['F01FD076']].rename(columns={'F01FD076':'sex'})
    # 1 is female, 0 is male
    sex.loc[sex['sex'] == 2, 'sex'] = 'Female'
    sex.loc[sex['sex'] == 1, 'sex'] = 'Male'

    death_idx = 7
    event = outcome[['F40FD025']].rename(columns={'F40FD025':'event'})
    event.loc[event['event'] != death_idx, 'event'] = 0
    event.loc[event['event'] == death_idx, 'event'] = 1

    # TODO: check if this days matches the actual date
    # TODO: where are the rest of the patients?
    time = ((time[['F22FD024']].rename(columns={'F22FD024':'time'}) - (dataset[['F01FD063']].rename(columns={'F01FD063':'time'}))) / 365.25).dropna()

    dataset = time.join(event, how='inner').join(sex, how='inner').join(treatment, how='inner')

    dataset['condition'] = [", ".join([s, i]) for s, i in dataset[['sex', 'intervention']].values]

    return dataset, 'condition'



def topcat():
    # hfhosp worked (kind of)
    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'TOPCAT/datasets'

    data_file = 't003.sas7bdat'
    outcome_file = 'outcomes.sas7bdat'

    data_path = os.path.join(directory, data_dir, data_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)

    dataset = pd.read_sas(data_path).set_index('ID')
    outcome = pd.read_sas(outcome_path).set_index('ID')

    sex = dataset[['GENDER']].rename(columns={'GENDER':'sex'})
    # 1 male, 2 female
    sex.loc[sex['sex'] == 2, 'sex'] = 'Female'
    sex.loc[sex['sex'] == 1, 'sex'] = 'Male'

    intervention = outcome[['drug']].rename(columns={'drug':'intervention'})
    # 1=Spironolactone, 2=Placebo
    intervention.loc[intervention['intervention'] == 1, 'intervention'] = 'Spironolactone'
    intervention.loc[intervention['intervention'] == 2, 'intervention'] = 'Placebo'

    e = 'stroke'
    t = 'time_stroke'
    outcome = outcome[[e, t]].rename(columns={e:'event', t:'time'})

    dataset = outcome.join(sex, how='inner').join(intervention, how='inner')

    dataset['condition'] = [", ".join([s, i]) for s, i in dataset[['sex', 'intervention']].values]

    return dataset, 'condition'

