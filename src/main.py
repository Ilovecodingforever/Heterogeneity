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

sys.path.append('../')
sys.path.append('../auton-survival')

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
from examples.cmhe_demo_utils import *



from data import topcat, accord, allhat, bari2d, sts



# dict = pd.read_csv('/zfsauton2/home/mingzhul/Heterogeneity/STS_preprocessing_files/que_DataDictionary.csv', encoding='latin1')
# not_found = []
# for i in range(len(data.columns)):
#     if data.columns[i].lower() not in dict['Group Of ShortName'].str.lower().values:
#         not_found.append(data.columns[i])





if __name__ == '__main__':
    dataset, attribute_var, model = bari2d()
    sts(model)


    f, axs = plt.subplots(1, 2, sharey=True)

    d = dataset.loc[dataset['intervention'] == 'Early Revascularization (treatment)']
    plot_kaplanmeier(d[['time', 'event']], groups=d['sex'], ax=axs[0])
    axs[0].set_xlabel('Time to Unstable Angina (years)')
    axs[0].set_ylabel('Survival probability')
    axs[0].set_title(dataset_name + ' Kaplan-Meier curve for Early Revascularization (treatment)')

    d = dataset.loc[dataset['intervention'] == 'Medical Therapy (control)']
    plot_kaplanmeier(d[['time', 'event']], groups=d['sex'], ax=axs[1])
    axs[1].set_xlabel('Time to Unstable Angina (years)')
    axs[1].set_ylabel('Survival probability')
    axs[1].set_title(dataset_name + ' Kaplan-Meier curve for Medical Therapy (control)')

    axs[0].grid(True)
    axs[1].grid(True)
    f.set_size_inches(14, 7)
    # plt.ylim(0, 1)

    plt.savefig('kaplanmeier.png')
    plt.close()


    f, axs = plt.subplots(1, 3, sharey=True)
    d = dataset.loc[dataset['sex'] == 'Female']
    plot_kaplanmeier(d[['time', 'event']], groups=d['intervention'], ax=axs[0])
    axs[0].set_xlabel('Time to Unstable Angina (years)')
    axs[0].set_ylabel('Survival probability')
    axs[0].set_title(dataset_name + ' Kaplan-Meier curve for Female')

    d = dataset.loc[dataset['sex'] == 'Male']
    plot_kaplanmeier(d[['time', 'event']], groups=d['intervention'], ax=axs[1])
    axs[1].set_xlabel('Time to Unstable Angina (years)')
    axs[1].set_ylabel('Survival probability')
    axs[1].set_title(dataset_name + ' Kaplan-Meier curve for Male')

    d = dataset
    plot_kaplanmeier(d[['time', 'event']], groups=d['intervention'], ax=axs[2])
    axs[2].set_xlabel('Time to Unstable Angina (years)')
    axs[2].set_ylabel('Survival probability')
    axs[2].set_title(dataset_name + ' Kaplan-Meier curve for Female and Male')

    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    f.set_size_inches(21, 7)
    # plt.ylim(0, 1)

    plt.savefig('kaplanmeier_sex.png')
    plt.close()

    print()



