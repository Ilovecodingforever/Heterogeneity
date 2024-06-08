import os
import sys
import copy
import itertools

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree

# sys.path.append('../')
# sys.path.append('../auton-survival')

import auton_survival
# from auton_survival.reporting import plot_kaplanmeier
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.metrics import phenotype_purity
from auton_survival.models.dcm import DeepCoxMixtures
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import brier_score
from auton_survival.models.dcm.dcm_utilities import predict_latent_z
from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects


from cmhe_demo_utils import *

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts




def print_tree(clf, preprocessor):

    mean = preprocessor.scaler.scaler.mean_
    std = np.sqrt(preprocessor.scaler.scaler.var_)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        
        # if node_depth[i] > 3:
        #     continue
        
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=values[i]
                )
            )
        else:
            thres = threshold[i]
            if features.columns[feature[i]] == 'age':
                thres = thres*std[0] + mean[0]
            if features.columns[feature[i]] == 'bmi':
                thres = thres*std[1] + mean[1]
            
            print(
                # "{space}node={node} is a split node with value={value}: "
                "{space}node={node}: "
                # "go to node {left} if X[:, {feature}] <= {threshold} "
                "go to node {left} if {feature_name} <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    # feature=feature[i],
                    feature_name=features.columns[feature[i]],
                    threshold=thres,
                    right=children_right[i],
                    value=values[i],
                )
            )
                



def plot_kaplanmeier(outcomes, groups=None, plot_counts=False, ax=None, **kwargs):

    """Plot a Kaplan-Meier Survival Estimator stratified by groups.

    Parameters
    ----------
    outcomes: pandas.DataFrame
        a pandas dataframe containing the survival outcomes. The index of the
        dataframe should be the same as the index of the features dataframe.
        Should contain a column named 'time' that contains the survival time and
        a column named 'event' that contains the censoring status.
        \( \delta_i = 1 \) if the event is observed.
    groups: pandas.Series
        a pandas series containing the groups to stratify the Kaplan-Meier
        estimates by.
    plot_counts: bool
        if True, plot the number of at risk and censored individuals in each group.

    """

    if groups is None:
        groups = np.array([1]*len(outcomes))

    curves = {}

    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    for group in sorted(set(groups)):
        if pd.isna(group): continue

        curves[group] = KaplanMeierFitter().fit(outcomes.loc[groups==group]['time'],
                                                outcomes.loc[groups==group]['event'])
        ax = curves[group].plot(label=group, ax=ax, **kwargs)

    if plot_counts:
        add_at_risk_counts(iter([curves[group] for group in curves]), ax=ax)

    return ax



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


def plot_KM(phenotypes, outcomes, features, condition_names, condition, treatment=None, name=''):
    '''
    phenotypes: the phenotypes of the individuals
    outcomes: the outcomes dataframe
    features: the features dataframe
    condition_names: dictionary of the feature used to stratify the KM curve
    treatment: the treatment column
    condition: name of the feature used to stratify the KM curve
    
    '''

    if treatment is None:
        f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 10))
        for j, c in enumerate(condition_names.keys()):
            groups = pd.DataFrame({'phenotypes': phenotypes+1, 'condition': features[condition]}, index=outcomes.index)
            d = groups[groups['condition'] == c]['phenotypes'].apply(lambda x: 'Phenotype ' + str(x))
            # groups = groups.apply(lambda x: str(condition_names[x['condition']]) + ' Phenotype ' + str(x['phenotypes']), axis=1)
            ax = plot_kaplanmeier(outcomes.loc[d.index], plot_counts=False, groups=d, ax=axs[j])

            ax.set_title(condition_names[c])
            ax.grid(True)

        plt.suptitle(name + ' Kaplan-Meier curve')
        plt.xlabel('Time to death (years)')
        plt.ylabel('Survival probability')
        # f.set_size_inches(14*2, 7*2)
        plt.savefig(name+'_KM.png')
        plt.close()


        f, ax = plt.subplots(1, figsize=(15, 10))
        groups = pd.DataFrame({'phenotypes': phenotypes+1}, index=outcomes.index)
        groups = groups.apply(lambda x: 'Phenotype ' + str(x['phenotypes']), axis=1)
        ax = plot_kaplanmeier(outcomes, plot_counts=False, groups=groups, ax=ax)        

        plt.suptitle(name + ' Kaplan-Meier curve')
        plt.xlabel('Time to death (years)')
        plt.ylabel('Survival probability')
        plt.grid(True)
        # f.set_size_inches(14, 7)
        plt.savefig(name+'_KM_all.png')
        plt.close()
        
        return


    f, axs = plt.subplots(len(condition_names.keys()), len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))

    treatment = treatment.apply(lambda x: 'Treatment' if x == 1 else 'Control')

    for i, p in enumerate(np.unique(phenotypes)):
        for j, c in enumerate(condition_names.keys()):
            d = outcomes.loc[features.index].loc[phenotypes==p]
            ax = plot_kaplanmeier(d.loc[features[condition] == c], treatment.loc[features[condition] == c], plot_counts=False, ax=axs[j, i])

            ax.set_xlabel('Time to death (years)')
            ax.set_ylabel('Survival probability')
            ax.set_title(condition_names[c] + ' Phenotype '+ str(i+1))
            ax.grid(True)

    plt.suptitle(name + ' Kaplan-Meier curve')
    # f.set_size_inches(14*2, 7*2)
    plt.savefig(name+'_KM.png')
    plt.close()


    f, axs = plt.subplots(1, len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))
    for i, p in enumerate(np.unique(phenotypes)):

        d = outcomes.loc[features.index].loc[phenotypes==p]
        ax = plot_kaplanmeier(d, treatment.loc[d.index], plot_counts=False, ax=axs[i])

        ax.set_xlabel('Time to death (years)')
        ax.set_ylabel('Survival probability')
        ax.set_title('Phenotype '+ str(i+1))
        ax.grid(True)

    # f.set_size_inches(14, 7)
    plt.suptitle(name + ' Kaplan-Meier curve')
    plt.savefig(name+'_KM_all.png')
    plt.close()

    print()





def fit_CMHE(x, t, e, a):

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
        model = DeepCoxMixturesHeterogenousEffects(random_seed=10, k=k, g=g, layers=layers)
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
    model = DeepCoxMixturesHeterogenousEffects(random_seed=10, k=k, g=g, layers=layers)
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




def plot_tree(features, phenotypes, name='', preprocessor=None):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, phenotypes)
    plt.figure(figsize=(50, 10))
    tree.plot_tree(clf, fontsize=10, max_depth=4, feature_names=features.columns,
                class_names=['Phenotype 1', 'Phenotype 2'], filled=True, rounded=True)
    plt.title(name + ' Decision Tree for Phenotypes')
    plt.savefig(name+'_tree.pdf')
    plt.close()

    print('depth of tree', clf.get_depth())

    return









def phenotyping(outcomes_raw, features_raw, treatment, cat_feats, num_feats, model=None, name=''):

    '''
    outcomes_raw: the outcomes dataframe
    features_raw: the features dataframe
    treatment: the treatment column
    cat_feats, num_feats: the features that need to be preprocessed (scaled, one-hot encoded)
    model: if None, train a new model, otherwise use the given model
    name: name of the dataset
    '''

    # Identify categorical (cat_feats) and continuous (num_feats) features
    features = features_raw
    outcomes = outcomes_raw.loc[features.index]

    # standardize, one hot encode
    x = features.copy()
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    # processed = preprocessor.fit_transform(x[cat_feats + num_feats], cat_feats=cat_feats, num_feats=num_feats,
    #                                 one_hot=True, fill_value=-1).astype(float)
    preprocessor = preprocessor.fit(x[cat_feats + num_feats], cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1)
    processed = preprocessor.transform(x[cat_feats + num_feats]).astype(float)    
    x.drop(cat_feats + num_feats, axis=1, inplace=True)
    x[processed.columns] = processed

    # train, or apply
    if model is None:
        phenotypes, model = fit_CMHE(x, outcomes['time'], outcomes['event'], treatment)
    else:
        phenotypes = predict_CMHE(model, x)

    # plot KM
    plot_KM(phenotypes, outcomes, features, {2: 'Female', 1: 'Male'}, 
            'female' if 'female' in cat_feats else 'sex',
            treatment=treatment, name=name)

    # plot tree
    plot_tree(x, phenotypes, name=name, preprocessor=preprocessor)

    return phenotypes, model

