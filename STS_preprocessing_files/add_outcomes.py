import pandas as pd
import yaml
import numpy as np
from utils import add_outcomes, relabel_outcome_feature, agg_multilabel_feature

config = yaml.load(open('./Data_Processing_config.yaml'), Loader=yaml.FullLoader)

df = pd.read_csv('./CABG_Cohort_Preopt_Data.csv')
df.set_index(config['ID'], inplace=True)

sts_outcome_df = pd.read_csv(config['cohort_data_path'])

outcome_df = add_outcomes(df, sts_outcome_df, config['ID'], config['class'])

outcome_df.to_csv('{}_Cohort_Preopt_Data_with_Outcomes.csv'.format(config['data_procedure']), index=False) 