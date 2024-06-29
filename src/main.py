"""
Not found:
Priorrev
Priorstent                              StentPrsnt
Hxcva                                   StrokeFlag(?), CNStrokP




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


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
import os
import sys
import copy
import itertools

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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




from data import bari2d, sts



# dict = pd.read_csv('/zfsauton2/home/mingzhul/Heterogeneity/STS_preprocessing_files/que_DataDictionary.csv', encoding='latin1')
# not_found = []
# for i in range(len(data.columns)):
#     if data.columns[i].lower() not in dict['Group Of ShortName'].str.lower().values:
#         not_found.append(data.columns[i])



def d():
    data_raw_sts = pd.read_csv('src/STS_preprocessing_files/timetoevent_cabg.csv')

    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'BARI2D/data'

    data_file = 'bari2d_bl.sas7bdat'
    outcome_file = 'bari2d_endpts.sas7bdat'
    # outcome_file = 'bari2d_long.sas7bdat'

    data_path = os.path.join(directory, data_dir, data_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)

    dataset_raw_bari2d = pd.read_sas(data_path).set_index('id')
    outcome = pd.read_sas(outcome_path).set_index('id')

    # intervention = dataset_raw['cardtrt'].rename('intervention')


    dataset_raw_bari2d['race'] = dataset_raw_bari2d['race'] == 1
    dataset_raw_bari2d['smkcat'] = dataset_raw_bari2d['smkcat'] != 0
    d_bari2d = dataset_raw_bari2d[['hxchl', 'ablvef', 'screat', 'hba1c', 'hxetoh', 'hxmi', 'hxchf', 'hxhtn', 'bmi', 'sex', 'smkcat', 'race', 'hispanic', 'age']]
    for c in d_bari2d.columns:
        print(c, d_bari2d[c].unique())


    cat = ['hxchl', 'ablvef', 'hxetoh', 'hxmi', 'hxchf', 'hxhtn', 'sex', 'smkcat', 'race', 'hispanic']
    num = ['screat', 'hba1c', 'bmi', 'age']

    data_raw_sts['female'] += 1      # Female: 2, Male: 1
    data_raw_sts['prevmi'] = np.clip(data_raw_sts['prevmi'], 0, 1)
    data_raw_sts['hypertn'] = np.clip(data_raw_sts['hypertn'], 0, 1)
    data_raw_sts['alcohol'] = data_raw_sts['alcohol'] > 0
    d_sts = data_raw_sts[['dyslip', 'creatlst', 'a1clvl', 'alcohol', 'prevmi', 'chf', 'hypertn', 'bmi', 'female', 'recentishsmoker', 'racecaucasian', 'ethnicity', 'age']]
    d_sts['ablvef'] = data_raw_sts['hdef'] < 50

    for c in d_sts.columns:
        print(c, d_sts[c].unique())

    cat = ['dyslip', 'alcohol', 'prevmi', 'chf', 'hypertn', 'female', 'recentishsmoker', 'racecaucasian', 'ethnicity']
    num = ['hdef', 'creatlst', 'a1clvl', 'bmi', 'age']




    # alcohol
    # female
    # recentishsmoker
    # race



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





if __name__ == '__main__':
    random_seed = 10

    # d()

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    bari2d_phenotypes, model = bari2d()
    sts_phenotypes, model = sts(model)

    print()


