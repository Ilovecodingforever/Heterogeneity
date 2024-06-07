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




from data import bari2d, sts, get_characteristics_bari2d_sts



# dict = pd.read_csv('/zfsauton2/home/mingzhul/Heterogeneity/STS_preprocessing_files/que_DataDictionary.csv', encoding='latin1')
# not_found = []
# for i in range(len(data.columns)):
#     if data.columns[i].lower() not in dict['Group Of ShortName'].str.lower().values:
#         not_found.append(data.columns[i])





if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    bari2d_phenotypes, model = bari2d()
    sts_phenotypes, model = sts(model)


    get_characteristics_bari2d_sts(bari2d_phenotypes, sts_phenotypes)
    
    print()



