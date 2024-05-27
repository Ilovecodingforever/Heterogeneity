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




def find_max_treatment_effect_phenotype(g, zeta_probs, factual_outcomes):
    """
    Find the group with the maximum treatement effect phenotype
    """
    mean_differential_survival = np.zeros(zeta_probs.shape[1]) # Area under treatment phenotype group
    outcomes_train, interventions_train = factual_outcomes

    # Assign each individual to their treatment phenotype group
    for gr in range(g): # For each treatment phenotype group
        # Probability of belonging the the g^th treatment phenotype
        zeta_probs_g = zeta_probs[:, gr]
        # Consider only those individuals who are in the top 75 percentiles in this phenotype
        z_mask = zeta_probs_g>np.quantile(zeta_probs_g, 0.75)

        mean_differential_survival[gr] = find_mean_differential_survival(
            outcomes_train.loc[z_mask], interventions_train.loc[z_mask])

    return np.nanargmax(mean_differential_survival)


def plot_KM(phenotypes, condition_, outcomes, features, condition_names, a_tr_names, condition,
            all_in_one=False, name=''):

    if all_in_one:
        f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 10))
        
        for j, c in enumerate(condition_):
            groups = pd.DataFrame({'phenotypes': phenotypes+1, 'condition': features[condition]}, index=outcomes.index)
            d = groups[groups['condition'] == c]['phenotypes'].apply(lambda x: 'Phenotype ' + str(x))
            # groups = groups.apply(lambda x: str(condition_names[x['condition']]) + ' Phenotype ' + str(x['phenotypes']), axis=1)
            plot_kaplanmeier(outcomes.loc[d.index], ax=axs[j], plot_counts=False, groups=d)

            axs[j].set_title(condition_names[c])
            axs[j].grid(True)

        plt.suptitle(name + ' Kaplan-Meier curve')
        plt.xlabel('Time to death (years)')
        plt.ylabel('Survival probability')
        # f.set_size_inches(14*2, 7*2)
        plt.savefig(name+'_KM.png')
        plt.close()


        f, ax = plt.subplots(1, figsize=(15, 10))
        groups = pd.DataFrame({'phenotypes': phenotypes+1}, index=outcomes.index)
        groups = groups.apply(lambda x: 'Phenotype ' + str(x['phenotypes']), axis=1)
        plot_kaplanmeier(outcomes, ax=ax, plot_counts=False, groups=groups)        

        plt.suptitle(name + ' Kaplan-Meier curve')
        plt.xlabel('Time to death (years)')
        plt.ylabel('Survival probability')
        plt.grid(True)
        # f.set_size_inches(14, 7)
        plt.savefig(name+'_KM_all.png')
        plt.close()
        
        return


    f, axs = plt.subplots(len(condition_), len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))

    a_tr_names = a_tr_names.apply(lambda x: 'Treatment' if x == 1 else 'Control')

    for i, p in enumerate(np.unique(phenotypes)):
        for j, c in enumerate(condition_):
            d = outcomes.loc[features.index].loc[phenotypes==p]
            plot_kaplanmeier(d[features[condition] == c], a_tr_names[features[condition] == c], ax=axs[j][i], plot_counts=False)

            axs[j][i].set_xlabel('Time to death (years)')
            axs[j][i].set_ylabel('Survival probability')
            axs[j][i].set_title(condition_names[c] + ' Phenotype '+ str(i+1))
            axs[j][i].grid(True)

    plt.suptitle(name + ' Kaplan-Meier curve')
    # f.set_size_inches(14*2, 7*2)
    plt.savefig(name+'_KM.png')
    plt.close()


    f, axs = plt.subplots(1, len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))
    for i, p in enumerate(np.unique(phenotypes)):

        d = outcomes.loc[features.index].loc[phenotypes==p]
        plot_kaplanmeier(d, a_tr_names.loc[d.index], ax=axs[i], plot_counts=False)

        axs[i].set_xlabel('Time to death (years)')
        axs[i].set_ylabel('Survival probability')
        axs[i].set_title('Phenotype '+ str(i+1))
        axs[i].grid(True)

    # f.set_size_inches(14, 7)
    plt.suptitle(name + ' Kaplan-Meier curve')
    plt.savefig(name+'_KM_all.png')
    plt.close()

    print()





def fit_CMHE(x, t, e, a):
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    horizons = [1, 3, 5]

    best_IBS = np.inf

    # Hyper-parameters to train model
    # ks = [1, 2, 3] # number of underlying base survival phenotypes
    # gs = [1, 2, 3] # number of underlying treatment effect phenotypes.
    # layerss = [[8, 8], [64, 64], [128, 128]] # number of neurons in each hidden layer.
    ks = [1] # number of underlying base survival phenotypes
    gs = [2] # number of underlying treatment effect phenotypes.
    layerss = [[50, 50]] # number of neurons in each hidden layer.

    iters = 100 # number of training epochs
    # learning_rate = 0.001
    learning_rate = 0.01
    batch_size = 256
    vsize = 0.15 # size of the validation split
    patience = 3
    optimizer = "Adam"

    # get val data
    x, t, e, a = x.to_numpy(), t.to_numpy(), e.to_numpy(), a.to_numpy()
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    x_tr, t_tr, e_tr, a_tr = x[idx], t[idx], e[idx], a[idx]
    vsize_ = int(vsize*x_tr.shape[0])
    x_vl, t_vl, e_vl, a_vl = x_tr[-vsize_:], t_tr[-vsize_:], e_tr[-vsize_:], a_tr[-vsize_:]
    x_tr, t_tr, e_tr, a_tr = x_tr[:-vsize_], t_tr[:-vsize_], e_tr[:-vsize_], a_tr[:-vsize_]


    # hyperparam search
    for k, g, layers in itertools.product(ks, gs, layerss):
        print(k, g, layers)
        # Instantiate the CMHE model
        model = DeepCoxMixturesHeterogenousEffects(random_seed=random_seed, k=k, g=g, layers=layers)
        model = model.fit(x_tr, t_tr, e_tr, a_tr, vsize=vsize, val_data=(x_vl, t_vl, e_vl, a_vl), iters=iters,
                        learning_rate=learning_rate, batch_size=batch_size,
                        optimizer=optimizer, patience=patience)
        print(f'Treatment Effect for the {g} groups: {model.torch_model[0].omega.detach()}')

        # Now let us predict survival using CMHE
        predictions_test_CMHE = model.predict_survival(x_vl, a_vl, t=horizons)
        CI1, CI3, CI5, IBS = factual_evaluate((x_tr, t_tr, e_tr, a_tr), (x_vl, t_vl, e_vl, a_vl),
                                            horizons, predictions_test_CMHE)
        print(f'IBS: {IBS}')
        if best_IBS > IBS:
            best_IBS = IBS
            best_params = (k, g, layers)


    print(f'Best IBS: {best_IBS}')
    print(f'Best Params: {best_params}')
    k, g, layers = best_params
    model = DeepCoxMixturesHeterogenousEffects(random_seed=random_seed, k=k, g=g, layers=layers)
    model = model.fit(x_tr, t_tr, e_tr, a_tr, vsize=vsize, val_data=(x_vl, t_vl, e_vl, a_vl), iters=iters,
                    learning_rate=learning_rate, batch_size=batch_size,
                    optimizer=optimizer, patience=patience)
    print(f'Treatment Effect for the {g} groups: {model.torch_model[0].omega.detach()}')

    zeta_train = predict_CMHE(model, x)
    
    return zeta_train, model



def predict_CMHE(model, x):

    zeta_probs_train = model.predict_latent_phi(x)
    zeta_train =  np.argmax(zeta_probs_train, axis=1)
    print(f'Distribution of individuals in each treatement phenotype in the training data: \
    {np.unique(zeta_train, return_counts=True)[1]}')
    
    return zeta_train




def phenotyping(outcomes_raw, features_raw, treatment, cat_feats, num_feats, model=None, all_in_one=False, name=''):

    '''
    cat_feats, num_feats: the features that need to be preprocessed (scaled, one-hot encoded)
    '''

    # Identify categorical (cat_feats) and continuous (num_feats) features
    features = features_raw
    outcomes = outcomes_raw.loc[features.index]

    x = features.copy()
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    processed = preprocessor.fit_transform(x[cat_feats + num_feats], cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1).astype(float)
    x.drop(cat_feats + num_feats, axis=1, inplace=True)
    x[processed.columns] = processed

    # data
    x_tr = x
    # outcomes
    y_tr = outcomes
    t_tr = outcomes['time']
    e_tr = outcomes['event']
    # treatment
    a_tr = treatment

    # train
    if model is None:
        phenotypes, model = fit_CMHE(x_tr, t_tr, e_tr, a_tr)
    else:
        phenotypes = predict_CMHE(model, x_tr)

    # graphs
    plot_KM(phenotypes, [1, 2], y_tr, features, {2: 'Female', 1: 'Male'},
            a_tr, 
            'female' if 'female' in cat_feats else 'sex',
            all_in_one=all_in_one, name=name)


    column_names = x_tr.columns

    # for p in np.unique(phenotypes):
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(features_raw[phenotypes == p][cat_feats+num_feats].describe())

    d = pd.get_dummies(features_raw[cat_feats+num_feats], columns=cat_feats, drop_first=True)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(d, phenotypes)
    plt.figure(figsize=(50, 10))
    tree.plot_tree(clf, fontsize=10, max_depth=4, feature_names=column_names,
                class_names=['Phenotype 1', 'Phenotype 2'], filled=True, rounded=True)
    plt.title(name + ' Decision Tree for Phenotypes')
    plt.savefig(name+'_tree.pdf')
    plt.close()

    print('depth of tree', clf.get_depth())

    return model

