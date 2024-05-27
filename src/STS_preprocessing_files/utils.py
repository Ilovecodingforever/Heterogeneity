import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def _ohe_feats(data, feature_name, index_name):
    ohe = OneHotEncoder(categories='auto', drop=None)
    ohe_feat = ohe.fit_transform(data[feature_name].values.reshape(-1, 1)).toarray()
    ohe_names = [i.replace('.0', '') for i in ohe.get_feature_names_out([feature_name])]
    ohe1 = pd.DataFrame(data=ohe_feat, columns=ohe_names).astype('int64')
    ohe1 = ohe1.iloc[:, :-1]
    ohe1.set_index(data.index.values, inplace=True)
    ohe1.index.name = index_name
    
    return ohe1

def add_outcomes(data, outcome_data, id_name, class_dict):
    for key, item in [feat for feat in class_dict.items()]:

        outcome = pd.DataFrame(outcome_data, 
                               columns = [id_name] + item).set_index(id_name)
        outcome = relabel_outcome_feature(outcome, class_dict)
        outcome = agg_multilabel_feature(outcome, key)
        outcome = outcome.astype('float64')

        outcome = outcome.loc[data.index.get_level_values(id_name)]
        data = pd.concat([data, outcome], axis=1)

    data.reset_index(drop=False, inplace=True)
    return data

def relabel_outcome_feature(outcome_df, class_dict):
    for feat in outcome_df.columns:
        # coprevlv feature has 3 and 4 value options
        if feat in class_dict['reoperation']:
            outcome_df.loc[outcome_df[feat] == 3, feat] = 1
            outcome_df.loc[outcome_df[feat] == 4, feat] = 1

        elif feat in class_dict['mortality']:
            outcome_df.loc[outcome_df[feat] == 1, feat] = 0
            outcome_df.loc[outcome_df[feat] == 2, feat] = 1
            outcome_df.loc[outcome_df[feat] == -9, feat] = np.nan

        elif feat in class_dict['stroke']:
            outcome_df.loc[outcome_df[feat] == 3, feat] = 1
            outcome_df.loc[outcome_df[feat] == 4, feat] = 1
            outcome_df.loc[outcome_df[feat] == 5, feat] = 1

        elif feat in class_dict['dswi']:
            outcome_df.loc[outcome_df[feat] == 3, feat] = 1
            outcome_df.loc[outcome_df[feat] == 4, feat] = 1

        else:
            continue
            
    return(outcome_df)

def agg_multilabel_feature(outcome_df, key):
    pos = (outcome_df.astype('float64').select_dtypes(include=['float64'])==1).any(axis=1)
    pos_ind = pos[pos==True].index.values.tolist()
    null = (outcome_df.astype('float64').select_dtypes(include=['float64']).isnull()==True).all(axis=1)
    null_ind = null[null==True].index.values.tolist()
    neg_ind = np.where(~np.in1d(outcome_df.index.values, pos_ind+null_ind))[0] 

    new_df = pd.DataFrame(pos)
    new_df.loc[pos_ind] = 1
    new_df.loc[null_ind] = np.nan
    new_df.iloc[neg_ind] = 0
    
    new_df.columns = [key]
    return new_df