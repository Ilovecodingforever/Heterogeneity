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
from create_figs import venn_diagram, histogram, km_curves


random_seed = 10

torch.manual_seed(random_seed)
np.random.seed(random_seed)

import random, os
import numpy as np
import torch

random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True    


if __name__ == '__main__':
    
    try: 
        if sys.argv[1] == 'macce':
            outcomes = ['macce']
        elif sys.argv[1] == 'both':
            outcomes = ['mortality', 'macce']
        else:
            outcomes = ['mortality']
    except IndexError:
        outcomes = ['mortality']

    for oc in outcomes:
        print ('Outcome being evaluated is: ' + oc)
        bari2d_phenotypes, model = bari2d(oc)
        
        sts_phenotypes, model = sts(model, oc)

        # create KM curves
        print(f'Saving KM curve for {oc}')
        km_curves(outcome = oc, save=True)

    # additional figures for publication
    print('Saving venn diagram and histogram figures')
    venn_diagram(save=True)
    histogram(save=True)
    print()





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



