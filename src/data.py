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



def get_characteristics(data, phenotype, cat_feats, num_feats, name):

    if name == 'STS':
        data['Operative_mortality'] = (data['mt30stat'] == 2) | (data['mtdcstat'] == 2)
        cat_feats += ['Operative_mortality']

    pd.set_option('display.max_rows', 500)
    assert(len(np.unique(phenotype) == 2))

    num_characteristics = data[num_feats].groupby(phenotype).describe().T
    mean_std = num_characteristics.loc[(slice(None), ['mean', 'std']), :].add_prefix('phenotype ')
    ttest = data[num_feats].apply(lambda x: scipy.stats.ttest_ind(x[phenotype == 0].dropna(), x[phenotype == 1].dropna(), equal_var=False).pvalue, axis=0).rename('p-value')
    mean_std.to_csv(f'characteristics/{name}_mean_std.csv')
    ttest.to_csv(f'characteristics/{name}_ttest.csv')
    # print('mean and std for numerical features:')
    # print(mean_std)
    # print('t-test for numerical features, p-values:')
    # print(ttest)

    lst = [pd.crosstab(data[feat], phenotype) for feat in cat_feats]
    # same:
    # lst = [(data[feat].groupby(phenotype).value_counts()).unstack(level=0) for feat in cat_feats]
    counts = pd.concat(lst, keys=cat_feats, names=['feature', 'value']).add_prefix('phenotype ')
    chisq = counts.groupby('feature').apply(lambda x: scipy.stats.chi2_contingency(x).pvalue).rename('p-value')
    # print('counts for categorical features:')
    # print(counts)
    counts.to_csv(f'characteristics/{name}_counts.csv')
    chisq.to_csv(f'characteristics/{name}_chisq.csv')
    # print('chi-squared test for categorical features, p-values:')
    # print(chisq)

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

    # TODO: smkcat
    dataset_raw['race'] = dataset_raw['race'] == 1
    dataset_raw['smkcat'] = dataset_raw['smkcat'] == 2
    cat_feats = ['hxchl', 'ablvef', 'hxetoh', 'hxmi', 'hxchf', 'hxhtn', 'sex',
                 'smkcat', 'race', 'hispanic']
    num_feats = ['screat', 'hba1c', 'bmi', 'age']

    phenotypes, model = phenotyping(outcome, dataset_raw.loc[outcome.index][cat_feats + num_feats],
                                    intervention.loc[outcome.index],
                                    cat_feats, num_feats, name='Bari2D')

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) <= 6]
    num_feats = [c for c in dataset_raw.columns if len(dataset_raw[c].unique()) > 6]
    print('Bari2D characteristics:')
    get_characteristics(dataset_raw.loc[outcome.index], phenotypes, cat_feats, num_feats, name='Bari2D')

    return phenotypes, model


"""
BARI2D:                                 STS:

Hxchl                                   dyslip
Ablvef (for STS, you have absolute LVEF, need to make it boolean, with EF <50%)   "hdef" <50
Screat                                  creatLst
Hba1c                                   a1cLvl
Hxetoh                                  alcohol
Hxmi                                    prevmi
Hxchf                                   chf
Hxhtn                                   hypertn
Bmi                                     bmi
Sex                                     female
Smkcat                                  recentishsmoker
Race? (non-white vs white)              racecaucasian
Hispanic?                               ethnicity
Age                                     age
"""


def sts(model=None, outcome='mortality'):
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
    # outcome_idx = 0
    if outcome == 'mortality':
        outcome_idx = 0
    elif outcome == 'mace':
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

    cat_feats = ['dyslip', 'hdef', 'alcohol', 'prevmi', 'chf', 'hypertn', 'female',
                 'recentishsmoker', 'racecaucasian', 'ethnicity']
    num_feats = ['creatlst', 'a1clvl', 'bmi', 'age']

    if model is None: 
        return


    phenotypes, model = phenotyping(outcome, dataset[cat_feats+num_feats], None, cat_feats, num_feats, model, name='STS')

    # checked that max # categories: 6
    # the features that need to be preprocessed (scaled, one-hot encoded)
    cat_feats = [c for c in dataset.columns if len(dataset[c].unique()) <= 6]
    num_feats = [c for c in dataset.columns if len(dataset[c].unique()) > 6]
    print('STS characteristics:')
    get_characteristics(dataset, phenotypes, cat_feats, num_feats, name='STS')

    return phenotypes, model


