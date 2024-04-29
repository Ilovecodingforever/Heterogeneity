import os
import sys

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree

sys.path.append('auton-survival')

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



def phenotyping(outcomes_raw, features_raw, condition='sex', intervention='cardtrt'):

    condition_ = pd.unique(features_raw[condition])
    phenotypes_lst = []

    # 0=Medical Therapy, 1=Early Revascularization
    # 1 male, 2 female
    condition_names = {1: 'Male', 2:  'Female'}
    intervention_names = {0: 'Medical Therapy (control)', 1: 'Early Revascularization (treatment)'}

    # Identify categorical (cat_feats) and continuous (num_feats) features
    cat_feats = ['sex',
        'race', 'hxmi', 'hxhtn',
                'angcls6w', 'smkcat', 'ablvef', 'geographic_reg'
                ]
    num_feats = ['age', 'bmi', 'dmdur'
                ]


    features = features_raw
    outcomes = outcomes_raw.loc[features.index]

    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean')
    x = preprocessor.fit_transform(features[cat_feats + num_feats], cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1).astype(float)

    x_tr = x
    y_tr = outcomes
    outcomes_tr = y_tr
    t_tr = outcomes['time']
    e_tr = outcomes['event']
    
    
    a_tr = features[intervention]
    interventions_tr = a_tr
    a_tr_names = a_tr.apply(lambda x: intervention_names[x])
    
    # a_tr = features[condition]
    # interventions_tr = a_tr
    # a_tr_names = a_tr.apply(lambda x: condition_names[x])
    
    
    sex_tr_names = features[condition].apply(lambda x: condition_names[x])
    condition_tr_names = pd.concat([a_tr_names, sex_tr_names], axis=1)

    # Hyper-parameters to train model
    k = 1 # number of underlying base survival phenotypes
    g = 2 # number of underlying treatment effect phenotypes.
    layers = [50, 50] # number of neurons in each hidden layer.

    random_seed = 10
    iters = 100 # number of training epochs
    learning_rate = 0.01
    batch_size = 256
    vsize = 0.15 # size of the validation split
    patience = 3
    optimizer = "Adam"

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Instantiate the CMHE model
    model = DeepCoxMixturesHeterogenousEffects(random_seed=random_seed, k=k, g=g, layers=layers)

    model = model.fit(x_tr, t_tr, e_tr, a_tr, vsize=vsize, val_data=None, iters=iters,
                    learning_rate=learning_rate, batch_size=batch_size,
                    optimizer=optimizer, patience=patience)

    print(f'Treatment Effect for the {g} groups: {model.torch_model[0].omega.detach()}')

    zeta_probs_train = model.predict_latent_phi(x_tr)
    zeta_train =  np.argmax(zeta_probs_train, axis=1)
    print(f'Distribution of individuals in each treatement phenotype in the training data: \
    {np.unique(zeta_train, return_counts=True)[1]}')

    phenotypes = zeta_train

    f, axs = plt.subplots(len(condition_), len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))

    for i, p in enumerate(np.unique(phenotypes)):
        for j, c in enumerate(condition_):
            d = outcomes.loc[x_tr.index].loc[phenotypes==p]
            # d['condition'] = [", ".join([s, i]) for s, i in condition_tr_names.loc[d.index][[condition, intervention]].values]
            
            # Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
            # plot_kaplanmeier(d, d['condition'], ax=axs[i], plot_counts=True)
            plot_kaplanmeier(d[features[condition] == c], a_tr_names[features[condition] == c], ax=axs[j][i], plot_counts=False)

            axs[j][i].set_xlabel('Time to Unstable Angina (years)')
            axs[j][i].set_ylabel('Survival probability')
            axs[j][i].set_title('Kaplan-Meier curve for ' + condition_names[c] + ' Phenotype '+ str(i+1))
            axs[j][i].grid(True)

    # f.set_size_inches(14*2, 7*2)
    plt.savefig('_KM.png')
    plt.close()
    
    
    f, axs = plt.subplots(1, len(np.unique(phenotypes)), sharey=True, figsize=(15, 10))
    for i, p in enumerate(np.unique(phenotypes)):
        
        d = outcomes.loc[x_tr.index].loc[phenotypes==p]
        
        # Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
        plot_kaplanmeier(d, a_tr_names.loc[d.index], ax=axs[i], plot_counts=False)

        axs[i].set_xlabel('Time to Unstable Angina (years)')
        axs[i].set_ylabel('Survival probability')
        axs[i].set_title('Kaplan-Meier curve for Phenotype '+ str(i+1))
        axs[i].grid(True)

    # f.set_size_inches(14, 7)
    plt.savefig('_KM_all.png')
    plt.close()    
    
    
    
    
    column_names = ['Age', 'BMI', 'Duration of Diabetes', 'Non-White', 'History of MI',
                    'History of Hypertension', 
                    'Baseline Stable 3, 4 Angina', 'Baseline Unstable Angina', 'Baseline No Angina',
                    'Former Smoker', 'Current Smoker',
                    'Abnormal LVEF', 'Non-USA']
    
    

    for s in condition_:
        for p in np.unique(phenotypes):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(features_raw[phenotypes == p][features_raw[condition] == s][cat_feats+num_feats].describe())

        d = pd.get_dummies(features_raw[features_raw[condition] == s][cat_feats+num_feats], columns=cat_feats, drop_first=True)
                
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(d, phenotypes[features_raw[condition] == s])
        plt.figure(figsize=(30, 10)) 
        tree.plot_tree(clf, fontsize=10, max_depth=3, feature_names=column_names, 
                    class_names=['Phenotype 1', 'Phenotype 2'], filled=True, rounded=True)
        plt.title('Decision Tree for ' + condition_names[s] + ' Phenotypes')
        plt.savefig('tree_'+str(s)+'.png')
        plt.close()



    for s in condition_:
        for p in np.unique(phenotypes):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(features_raw[phenotypes == p][features_raw[condition] == s][cat_feats+num_feats].describe())

        d = pd.get_dummies(features_raw[features_raw[condition] == s][cat_feats+num_feats], columns=cat_feats, drop_first=True)
                
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(d, phenotypes[features_raw[condition] == s])
        plt.figure(figsize=(25*5, 10*5)) 
        tree.plot_tree(clf, fontsize=10, feature_names=column_names, 
                    class_names=['Phenotype 1', 'Phenotype 2'], filled=True, rounded=True)
        plt.title('Decision Tree for Phenotypes')
        plt.savefig('example.png')
        plt.close()





    # for c in condition_:
    #     features = features_raw[features_raw[condition] == c]
    #     outcomes = outcomes_raw.loc[features.index]

    #     preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean')
    #     x = preprocessor.fit_transform(features[cat_feats + num_feats], cat_feats=cat_feats, num_feats=num_feats,
    #                                     one_hot=True, fill_value=-1).astype(float)

    #     x_tr = x
    #     y_tr = outcomes
    #     outcomes_tr = y_tr
    #     t_tr = outcomes['time']
    #     e_tr = outcomes['event']
    #     a_tr = features[intervention]
    #     interventions_tr = a_tr
    #     a_tr_names = a_tr.apply(lambda x: intervention_names[x])

    #     # Hyper-parameters to train model
    #     k = 1 # number of underlying base survival phenotypes
    #     g = 2 # number of underlying treatment effect phenotypes.
    #     layers = [50, 50] # number of neurons in each hidden layer.

    #     random_seed = 10
    #     iters = 100 # number of training epochs
    #     learning_rate = 0.01
    #     batch_size = 256
    #     vsize = 0.15 # size of the validation split
    #     patience = 3
    #     optimizer = "Adam"

    #     torch.manual_seed(random_seed)
    #     np.random.seed(random_seed)

    #     # Instantiate the CMHE model
    #     model = DeepCoxMixturesHeterogenousEffects(random_seed=random_seed, k=k, g=g, layers=layers)

    #     model = model.fit(x_tr, t_tr, e_tr, a_tr, vsize=vsize, val_data=None, iters=iters,
    #                     learning_rate=learning_rate, batch_size=batch_size,
    #                     optimizer=optimizer, patience=patience)

    #     print(f'Treatment Effect for the {g} groups: {model.torch_model[0].omega.detach()}')

    #     zeta_probs_train = model.predict_latent_phi(x_tr)
    #     zeta_train =  np.argmax(zeta_probs_train, axis=1)
    #     print(f'Distribution of individuals in each treatement phenotype in the training data: \
    #     {np.unique(zeta_train, return_counts=True)[1]}')

    #     # max_treat_idx_CMHE = find_max_treatment_effect_phenotype(
    #     #     g=2, zeta_probs=zeta_probs_train, factual_outcomes=(outcomes_tr, interventions_tr))
    #     # print(f'\nGroup {max_treat_idx_CMHE} has the maximum restricted mean survival time on the training data!')


    #     phenotypes = zeta_train


    #     f, axs = plt.subplots(1, len(np.unique(phenotypes)), sharey=True)

    #     for i, p in enumerate(np.unique(phenotypes)):
            
    #         d = outcomes.loc[x_tr.index].loc[phenotypes==p]
    #         # Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
    #         plot_kaplanmeier(d, a_tr_names.loc[d.index], ax=axs[i])

    #         axs[i].set_xlabel('Time to Unstable Angina (years)')
    #         axs[i].set_ylabel('Survival probability')
    #         axs[i].set_title('Kaplan-Meier curve for ' + condition_names[c] + ', Phenotype '+ str(i+1))
    #         axs[i].grid(True)

    #     f.set_size_inches(14, 7)
    #     plt.savefig('KM_'+str(condition_names[c])+'.png')
    #     plt.close()



    #     # Estimate the Integrated Brier Score at event horizons of 1, 2 and 5 years
    #     metric = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_tr,
    #                                     phenotypes_test=None, outcomes_test=None,
    #                                     strategy='instantaneous', horizons=[2, 3, 5],
    #                                     bootstrap=None)

    #     print(f'Phenotyping purity for event horizon of 2 year: {metric[0]} | 3 years: {metric[1]} | 5 years: {metric[2]}')

    #     phenotypes_lst.append(phenotypes)

    # for c, phenotypes in enumerate(phenotypes_lst):
    #     for p in np.unique(phenotypes):
    #         with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #             print(features_raw[features_raw[condition]==condition_[c]][phenotypes == p][cat_feats+num_feats].describe())

    #     d = pd.get_dummies(features_raw[features_raw[condition]==condition_[c]][cat_feats+num_feats], 
    #                    columns=cat_feats, drop_first=True)
                
    #     clf = tree.DecisionTreeClassifier()
    #     clf = clf.fit(d, phenotypes)
    #     plt.figure(figsize=(25, 10)) 
    #     tree.plot_tree(clf, fontsize=10, max_depth=3, feature_names=d.columns, class_names=['Phenotype 1', 'Phenotype 2'], filled=True, rounded=True)
    #     plt.title('Decision Tree for ' + condition_names[condition_[c]] + ' Phenotypes')
    #     plt.savefig('tree_'+str(condition_names[condition_[c]])+'.png')
    #     plt.close()

    return phenotypes_lst










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



def bari2d():
    # https://www.sciencedirect.com/science/article/pii/S0735109713007961
    # Angina worked?

    directory = '/zfsauton/project/public/chiragn/counterfactual_phenotyping/datasets'

    data_dir = 'BARI2D/data'

    data_file = 'bari2d_bl.sas7bdat'
    # outcome_file = 'bari2d_endpts.sas7bdat'
    outcome_file = 'bari2d_long.sas7bdat'

    data_path = os.path.join(directory, data_dir, data_file)
    outcome_path = os.path.join(directory, data_dir, outcome_file)

    dataset_raw = pd.read_sas(data_path).set_index('id')
    outcome = pd.read_sas(outcome_path).set_index('id')

    intervention = dataset_raw[['cardtrt']].rename(columns={'cardtrt':'intervention'})
    # 0=Medical Therapy, 1=Early Revascularization
    intervention.loc[intervention['intervention'] == 0, 'intervention'] = 'Medical Therapy (control)'
    intervention.loc[intervention['intervention'] == 1, 'intervention'] = 'Early Revascularization (treatment)'

    sex = dataset_raw[['sex']]
    # 1 male, 2 female
    sex.loc[sex['sex'] == 2, 'sex'] = 'Female'
    sex.loc[sex['sex'] == 1, 'sex'] = 'Male'

    # e ='death'
    # t = 'deathfu'
    # outcome = outcome[[e, t]].rename(columns={e:'event', t:'time'})

    # Angina
    outcome = outcome[['dayvisit', 'angcls']].rename(columns={'dayvisit':'time', 'angcls':'event'})

    outcome['event'][outcome['event'] != 3] = 0
    outcome['event'][outcome['event'] == 3] = 1
    outcome['time'] /= 365.25

    o_ = outcome.set_index('time', append=True)
    idx = o_['event'] == 1
    have_outcome = o_.loc[idx].reset_index().groupby('id').min('time')

    idx = o_['event'] == 0
    no_outcome = o_.loc[idx].reset_index().groupby('id').max('time')

    outcome_ = no_outcome.copy()
    outcome_.loc[have_outcome.index] = have_outcome
    outcome_ = outcome_.reset_index().set_index('id')

    dataset = outcome_.join(sex, how='inner').join(intervention, how='inner')

    dataset['condition'] = [", ".join([s, i]) for s, i in dataset[['sex', 'intervention']].values]



    phenotyping(outcome_, dataset_raw.loc[outcome_.index])



    return dataset, 'condition'






if __name__ == '__main__':

    dataset_name = 'BARI2D'

    if dataset_name == 'TOPCAT':
        dataset, attribute_var = topcat()
    elif dataset_name == 'ACCORD':
        dataset, attribute_var = accord()
    elif dataset_name == 'ALLHAT':
        dataset, attribute_var = allhat()
    elif dataset_name == 'BARI2D':
        dataset, attribute_var = bari2d()


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



