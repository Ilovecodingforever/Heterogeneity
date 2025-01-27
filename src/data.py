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








def bari2d_sts_characteristics(bari2d_data, sts_data, ):

    '''
    name, bari2d, sts:

    Congestive heart failure, hxchf, chf,
    ejection fraction < 50%, ablvef,  hdef,
    Hispanic, hispanic, ethnicity,
    Hyperlipedemia, hxchl, dyslip,
    Hypertension, hxhtn, hypertn,
    Myocardial infarction, hxmi,    prevmi,
    Race (white), race, racecaucasian,
    Recent alcohol use, hxetoh, alcohol,
    Recent smoker, smkcat,  recentishsmoker,
    Sex (female), sex,   female,

    Age (years), age, age
    Body mass index, bmi, bmi
    Creatinine (mg/dL), screat, creatLst
    Hemoglobin A1c (%), hba1c , a1cLvl
    sts predicted mortality
    operative mortality
    '''

    bari2d_data['data'] = 'Bari2d'
    sts_data['data'] = 'STS'

    sts_data['Operative_mortality'] = (sts_data['mt30stat'] == 2) | (sts_data['mtdcstat'] == 2)

    # rename columns to descriptive names
    bari2d_data.rename(columns = {'hxchf': 'Congestive heart failure',
                                    'ablvef': 'Ejection fraction < 50%',
                                    'hispanic': 'Hispanic',
                                    'hxchl': 'Hyperlipedemia',
                                    'hxhtn': 'Hypertension',
                                    'hxmi': 'Myocardial infarction',
                                    'race': 'Race (white)',
                                    'hxetoh': 'Recent alcohol use',
                                    'smkcat': 'Recent smoker',
                                    'sex': 'Sex (female)',
                                    'age': 'Age (years)',
                                    'bmi': 'Body mass index',
                                    'screat': 'Creatinine (mg/dL)',
                                    'hba1c': 'Hemoglobin A1c (%)',}, inplace = True)
    # rename columns to descriptive names
    sts_data.rename(columns = {'chf': 'Congestive heart failure',
                                    'hdef': 'Ejection fraction < 50%',
                                    'ethnicity': 'Hispanic',
                                    'dyslip': 'Hyperlipedemia',
                                    'hypertn': 'Hypertension',
                                    'prevmi': 'Myocardial infarction',
                                    'racecaucasian': 'Race (white)',
                                    'alcohol': 'Recent alcohol use',
                                    'recentishsmoker': 'Recent smoker',
                                    'female': 'Sex (female)',
                                    'age': 'Age (years)',
                                    'bmi': 'Body mass index',
                                    'creatlst': 'Creatinine (mg/dL)',
                                    'a1clvl': 'Hemoglobin A1c (%)',}, inplace = True)
    # combine data
    cat_feats= ['Congestive heart failure',
                'Ejection fraction < 50%',
                'Hispanic',
                'Hyperlipedemia',
                'Hypertension',
                'Myocardial infarction',
                'Race (white)',
                'Recent alcohol use',
                'Recent smoker',
                'Sex (female)',]
    num_feats = ['Age (years)',
                'Body mass index',
                'Creatinine (mg/dL)',
                'Hemoglobin A1c (%)',]
    data = pd.concat([bari2d_data[num_feats+cat_feats+['data']],
                      sts_data[num_feats+cat_feats+['data']]], axis = 0).set_index('data', append = True)

    pd.set_option('display.max_rows', 500)

    num_characteristics = data[num_feats].groupby('data').describe().T
    mean_std = num_characteristics.loc[(slice(None), ['mean', 'std']), :]
    ttest = data[num_feats].apply(lambda x: scipy.stats.ttest_ind(x.xs('Bari2d', level=1, drop_level=False).dropna(),
                                                                  x.xs('STS', level=1, drop_level=False).dropna(),
                                                                  equal_var=False).pvalue, axis=0).rename('p-value')

    lst = [pd.crosstab(data[feat], data.index.get_level_values(1)) for feat in cat_feats]
    counts = pd.concat(lst, keys=cat_feats, names=['feature', 'value'])
    chisq = counts.groupby('feature').apply(lambda x: scipy.stats.chi2_contingency(x).pvalue).rename('p-value')

    # Create numerical table for use in manuscript
    num_table = ttest.copy()
    mean_std.reset_index(level=1, inplace=True)
    mean = mean_std.loc[mean_std.iloc[:,0] == 'mean']
    mean = mean.drop(columns = mean.columns[0])
    mean.columns = mean.columns + '_mean'
    std = mean_std.loc[mean_std.iloc[:,0] == 'std']
    std = std.drop(columns = std.columns[0])
    std.columns = std.columns + '_std'
    num_table = pd.concat([num_table, mean, std], axis=1)
    num_table.index.name = 'feature'

    # add predicted mortality
    pred_mort = sts_data['PREDMORT']
    row = pd.Series({'STS_mean': pred_mort.mean(),
                     'STS_std': pred_mort.std(),
    }, name='STS predicted mortality')
    num_table = num_table.append(row)

    num_table.to_csv(f'./characteristics/Bari2d_STS_num_table.csv')

    # Create categorical table for use in manuscript
    cat_table = counts.copy().reset_index(level=1)
    cat_table['total (Bari2d + STS)'] = cat_table['Bari2d'] + cat_table['STS']
    cat_table['percent (STS / (Bari2d + STS))'] = 100 * cat_table['STS'] / cat_table['total (Bari2d + STS)']
    cat_table['p-value'] = cat_table.index
    cat_table['p-value'] = cat_table['p-value'].map(chisq).fillna(cat_table['p-value'])
    cat_table.sort_index(inplace=True)

    # add STS Operative mortality
    row = pd.Series({'value': 0,
                     'STS': len(sts_data['Operative_mortality'].dropna()) - sts_data['Operative_mortality'].sum(),
                     }, name='Operative mortality')
    cat_table = cat_table.append(row)
    row = pd.Series({'value': 1,
                        'STS': sts_data['Operative_mortality'].sum(),
                        }, name='Operative mortality')
    cat_table = cat_table.append(row)

    cat_table.to_csv(f'./characteristics/Bari2d_STS_cat_table.csv')
    ### end ###


def get_characteristics(data, phenotype, cat_feats, num_feats, name, oc=''):

    if name == 'STS':
        data['Operative_mortality'] = (data['mt30stat'] == 2) | (data['mtdcstat'] == 2)
        cat_feats += ['Operative_mortality']

    pd.set_option('display.max_rows', 500)
    assert(len(np.unique(phenotype) == 2))

    num_characteristics = data[num_feats].groupby(phenotype).describe().T
    mean_std = num_characteristics.loc[(slice(None), ['mean', 'std']), :].add_prefix('phenotype_')
    ttest = data[num_feats].apply(lambda x: scipy.stats.ttest_ind(x[phenotype == 0].dropna(), x[phenotype == 1].dropna(), equal_var=False).pvalue, axis=0).rename('p-value')
    mean_std.to_csv(f'./characteristics/{name}_mean_std_{oc}.csv')
    ttest.to_csv(f'./characteristics/{name}_ttest_{oc}.csv')
    # print('mean and std for numerical features:')
    # print(mean_std)
    # print('t-test for numerical features, p-values:')
    # print(ttest)

    lst = [pd.crosstab(data[feat], phenotype) for feat in cat_feats]
    # same:
    # lst = [(data[feat].groupby(phenotype).value_counts()).unstack(level=0) for feat in cat_feats]
    counts = pd.concat(lst, keys=cat_feats, names=['feature', 'value']).add_prefix('phenotype_')
    chisq = counts.groupby('feature').apply(lambda x: scipy.stats.chi2_contingency(x).pvalue).rename('p-value')
    # print('counts for categorical features:')
    # print(counts)

    counts.to_csv(f'./characteristics/{name}_counts_{oc}.csv')
    chisq.to_csv(f'./characteristics/{name}_chisq_{oc}.csv')
    # print('chi-squared test for categorical features, p-values:')
    # print(chisq)

    ### Keith added to make table easier to use

    # Create numerical table for use in manuscript
    num_table = ttest.copy()
    mean_std.reset_index(level=1, inplace=True)
    mean = mean_std.loc[mean_std.iloc[:,0] == 'mean']
    mean = mean.drop(columns = mean.columns[0])
    mean.columns = mean.columns + '_mean'
    std = mean_std.loc[mean_std.iloc[:,0] == 'std']
    std = std.drop(columns = std.columns[0])
    std.columns = std.columns + '_std'
    num_table = pd.concat([num_table, mean, std], axis=1)
    num_table.index.name = 'feature'
    num_table.to_csv(f'./characteristics/{name}_num_table_{oc}.csv')

    # Create categorical table for use in manuscript
    cat_table = counts.copy().reset_index(level=1)
    cat_table['total'] = cat_table['phenotype_0'] + cat_table['phenotype_1']
    cat_table['percent'] = 100 * cat_table['phenotype_1'] / cat_table['total']
    cat_table['p-value'] = cat_table.index
    cat_table['p-value'] = cat_table['p-value'].map(chisq).fillna(cat_table['p-value'])
    cat_table.sort_index(inplace=True)
    cat_table.to_csv(f'./characteristics/{name}_cat_table_{oc}.csv')
    ### end ###

    return




def bari2d(oc = 'mortality'):
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

    # dataset_raw = dataset_raw[dataset_raw['strata'] == b'CABG']
    # outcome = outcome.loc[dataset_raw.index]

    intervention = dataset_raw['cardtrt'].rename('intervention')
    # 0=Medical Therapy, 1=Early Revascularization
    # intervention.loc[intervention['intervention'] == 0, 'intervention'] = 'Medical Therapy (control)'
    # intervention.loc[intervention['intervention'] == 1, 'intervention'] = 'Early Revascularization (treatment)'

    # sex = dataset_raw[['sex']]
    # 1 male, 2 female
    # sex.loc[sex['sex'] == 2, 'sex'] = 'Female'
    # sex.loc[sex['sex'] == 1, 'sex'] = 'Male'
    if oc == 'mortality':
        e ='death'
        t = 'deathfu'
    elif oc == 'macce':
        e = 'dthmistr'
        t = 'dthmistrfu'
    else:
        raise ValueError('Patient event oucome not recorded properly, please select "mortality" or "macce".')

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

    # TODO: smkcat
    dataset_raw['race'] = dataset_raw['race'] == 1
    dataset_raw['smkcat'] = dataset_raw['smkcat'] == 2

    dataset_raw['ablvef'] = dataset_raw['ablvef'] == 1
    dataset_raw['hxetoh'] = dataset_raw['hxetoh'] == 1

    cat_feats = ['hxchl', 'ablvef', 'hxetoh', 'hxmi', 'hxchf', 'hxhtn', 'sex',
                 'smkcat', 'race', 'hispanic']
    num_feats = ['screat', 'hba1c', 'bmi', 'age']

    phenotypes, model = phenotyping(outcome, dataset_raw.loc[outcome.index][cat_feats + num_feats],
                                    intervention.loc[outcome.index],
                                    cat_feats, num_feats, name='Bari2D', oc=oc)

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) <= 6]
    num_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) > 6]
    get_characteristics(dataset_raw.loc[outcome.index], phenotypes, cat_feats, num_feats, name='Bari2D', oc=oc)

    return phenotypes, model, dataset_raw.loc[outcome.index]



def sts(model=None, oc='mortality'):
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
    if oc == 'mortality':
        outcome_idx = 0
    elif oc == 'macce':
        # MACCEFlag, MACCEReadmitDays
        outcome_idx = 12
    else:
        raise ValueError('Invalid outcome')

    outcome = outcome[[outcome_names[outcome_idx*2], outcome_names[outcome_idx*2+1]]].rename(
        columns={outcome_names[outcome_idx*2]: 'event',
                 outcome_names[outcome_idx*2+1]: 'time'})
    outcome = outcome.dropna()
    outcome['time'] /= 365.25

    dataset = data.drop(outcome_names, axis=1).set_index('ID').loc[outcome.index]

    dataset['female'] += 1      # Female: 2, Male: 1
    dataset['prevmi'] = np.clip(dataset['prevmi'], 0, 1)
    dataset['hypertn'] = np.clip(dataset['hypertn'], 0, 1)
    dataset['alcohol'] = dataset['alcohol'] > 0
    dataset['hdef'] = dataset['hdef'] < 50
    dataset['recentishsmoker'] = dataset['recentishsmoker'] == 1
    dataset['racecaucasian'] = dataset['racecaucasian'] == 1

    cat_feats = ['dyslip', 'hdef', 'alcohol', 'prevmi', 'chf', 'hypertn', 'female',
                 'recentishsmoker', 'racecaucasian', 'ethnicity']
    num_feats = ['creatlst', 'a1clvl', 'bmi', 'age']

    if model is None:
        return

    phenotypes, model = phenotyping(outcome, dataset[cat_feats+num_feats], None, cat_feats, num_feats, model, name='STS', oc = oc)

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset.columns if len(dataset[c].unique()) <= 6]
    num_feats = [c for c in dataset.columns if len(dataset[c].unique()) > 6]
    print('STS characteristics:')
    get_characteristics(dataset, phenotypes, cat_feats, num_feats, name='STS', oc=oc)

    return phenotypes, model, dataset


