import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder

config = yaml.load(open('Data_Processing_config.yaml'), Loader=yaml.FullLoader)

def _ohe_feats(data, feature_name, index_name):
    ohe = OneHotEncoder(categories='auto', drop=None)
    ohe_feat = ohe.fit_transform(data[feature_name].values.reshape(-1, 1)).toarray()
    ohe_names = [i.replace('.0', '') for i in ohe.get_feature_names_out([feature_name])]
    ohe1 = pd.DataFrame(data=ohe_feat, columns=ohe_names).astype('int64')
    ohe1 = ohe1.iloc[:, :-1]
    ohe1.set_index(data.index.values, inplace=True)
    ohe1.index.name = index_name
    
    return ohe1

demo = pd.read_csv(config['cohort_data_path'])
preopt = pd.DataFrame(demo, columns=['dID',
                                     'CREATLST', # creatinine level closest to the date and time prior surgery 
                                     'AGE', 
                                     'FEMALE',
                                     'racecaucasian', 
                                     'raceblack', 
                                     'raceasian',
                                     'racenativeam', 
                                     'BSA',
                                     'BMI',
                                     'HCT', # pre-op hemocrit
                                     'WBC', # pre-op white blood cell count
                                     'PLATELETS', # pre-op platelet count
                                     'pvd', #history of peripheral arterial disease
                                     'cva', 
                                     'cvawhen',
                                     'cvd', # history of cardiovascular disease
                                     'cvdtia', # history of a Transient Ischemic Attack (TIA). 
                                     'dialysis', 
                                     'carshock',
                                     'priorhf',
                                     'chf',
                                     'MEDADPIDIS', # number of days prior to surgery ADP Inhibitor use was discontinued
                                     'medacei48', # Indicate whether the patient received ACE or ARB Inhibitors within 24 hours preceding surgery
                                     'hypertn', # curret diagnosis of hypotension
                                     'immsupp', # immunosyppressive medication therapy within 30 days
                                     'medinotr', # inotropic agents within 48 hrs preceding surgery
                                     'iabpwhen', #preop, intraop, postop
                                     'ecmowhen', #preop, intraop, postop
                                     'cathbasassist', # whether the patient was placed on a catheter based assist device
                                     'prcab', # previous CABG prior to current admission
                                     'prvalve', # previous valve (before admission?)
                                     'ivdrugab', # illicit drug use
                                     'cvdcarsten', # which carotid artery was determined from any diagnostic test to be >= 50% stenotic.
                                     'vdinsufa', #aortic valve insufficiency
                                     'vdinsufm', #mitral valve insufficiency
                                     'vdinsuft', #Triscuspid valve insufficiency
                                     'arrhythmia', #history of arrhythemia before operative procedure
                                     'arrhythsss',
                                     'arrhythatrfib', # Indicate whether arrhythmia was afib
                                     'arrhythafib', # Indicate if afib and if so what type
                                     'arrhythaflutter',
                                     'arrhyththird',
                                     'arrhythsecond', # second degree heart block
                                     'arrhythvv',
                                     'infendty', # Type of endocarditis
                                     'chrlungd', # chronic lung disease
                                     'alcohol',
                                     'pneumonia', # history of pneumonia 
                                     'mediastrad', # history of mediastinal radiation
                                     'cancer', 
                                     'diabetes', 
                                     'diabctrl', # diabetes control method
                                     'incidenc', # which surgery number
                                     'numdisv', #num of diseased vessels
                                     'miwhen', # time b/w last MI and surgery
                                     'cardsymptimeofadm', # symps at this admission
                                     'tobaccouse', 
                                     'fhcad', #family history 
                                     'slpapn', # sleep apnea
                                     'liverdis', 
                                     'unrespstat', #history of unresponsive status
                                     'poc', #previous caridac intervention
                                     'status', # patient status before entering operating room
                                     'syncope',
                                     'vdstena', # aortic stenosis present
                                     'vdstenm', # mitral stenosis present
                                     'hmo2', # Indicate whether supplemental oxygen at home is prescribed and used.
                                     'PROCAICD', # Indicate whether the patient had a previous implant of an Implantable Cardioverter/Defibrillator.
                                     'pocpci', # Indicate whether a previous Percutaneous Coronary Intervention (PCI) was performed any time prior to this surgical procedure. 
                                     'pocpciwhen', # Indicate whether the previous Percutaneous Cardiac Intervention (PCI) was performed within this episode of care.
                                     'pocpciin', # Indicate the interval of time between the previous PCI and the current surgical procedure.
                                     'HDEF', # Indicate the percentage of the blood emptied from the left ventricle at the end of the contraction. Use the most recent determination prior to the surgical intervention
                                     'LMAINDIS', # Indicate whether the patient has Left Main Coronary Disease.
                                     'PROXLAD', # Indicate whether the percent luminal narrowing of the proximal left anterior descending artery at the point of maximal stenosis is greater than or equal to 70%.
                                     'medster', # Indicate whether the patient was taking steroids within 24 hours of surgery.
                                     'medgp', # Indicate whether the patient received Glycoprotein IIb/IIIa inhibitors within 24 hours preceding surgery.
                                     'payorprim', # Indicate the primary insurance payor for this admission.
                                     'PREDRENF', # STS Calculator: risk of renal failure
                                     'PREDMORT', # STS Calculator: risk of mortality
                                     'PREDVENT', # STS Calculator: risk of prolonged ventilation
                                     'PREDREOP', # STS Calculator: risk of reoperation
                                     'PREDSTRO', # STS Calculator: risk of stroke
                                     'PREDMM', # STS Calculator: risk of combined morbidity and mortality
                                     'PREDDEEP', # STS Calculator: risk of deep sternal wound infection
                                     'PRED6D', # STS Calculator: risk of 6 day readmission
                                     'PRED14D' # STS Calculator: risk of 14 day readmission
                                     ])


preopt.loc[preopt.HDEF>50, 'HDEF'] = 50

preopt.loc[preopt.BSA<1.4, 'BSA'] = 1.4
preopt.loc[preopt.BSA>2.6, 'BSA'] = 2.6

preopt.loc[preopt.BMI<18, 'BMI'] = 18
preopt.loc[preopt.BMI>50, 'BMI'] = 50

preopt.loc[preopt.dialysis == -9, 'dialysis'] = np.nan
preopt.loc[preopt.CREATLST<0.5, 'CREATLST'] = 0.5
preopt.loc[preopt.CREATLST>5.0, 'CREATLST'] = 5.0

preopt.loc[preopt.HCT<25, 'HCT'] = 25
preopt.loc[preopt.HCT>50, 'HCT'] = 50

preopt.loc[preopt.WBC<3.6, 'WBC'] = 3.6
preopt.loc[preopt.WBC>18, 'WBC'] = 18

preopt.loc[preopt.PLATELETS<80000, 'PLATELETS'] = 80000
preopt.loc[preopt.PLATELETS>425000, 'PLATELETS'] = 425000

preopt.loc[preopt.hypertn == -9, 'hypertn'] = np.nan

preopt.loc[preopt.immsupp == -9, 'immsupp'] = np.nan

preopt.loc[preopt.medster == -9, 'medster'] = np.nan
preopt.loc[preopt.medster == 3, 'medster'] = 0 # contraindicated/not indicated

preopt.loc[preopt.medgp == -9, 'medgp'] = np.nan
preopt.loc[preopt.medgp == 3, 'medgp'] = 0 # contraindicated/not indicated

preopt.loc[preopt.medinotr == 3, 'medinotr'] = 0 # contraindicated/not indicated

preopt.loc[preopt.iabpwhen == 2, 'iabpwhen'] = 0
preopt.loc[preopt.iabpwhen == 3, 'iabpwhen'] = 0

preopt.loc[preopt.ecmowhen == 2, 'ecmowhen'] = 0
preopt.loc[preopt.ecmowhen == 3, 'ecmowhen'] = 0
preopt.loc[preopt.ecmowhen == 4, 'ecmowhen'] = 0

preopt.loc[preopt.carshock == 1, 'carshock'] = 0
preopt.loc[preopt.carshock == 4, 'carshock'] = 0

ones = preopt.loc[lambda x: (preopt['ecmowhen']==1) | (preopt['carshock']==3) | (preopt['cathbasassist']==1)].index.values
zeros = preopt.loc[lambda x: (preopt['ecmowhen']==0) & (preopt['carshock']==0) & (preopt['cathbasassist']==0)].index.values
feat = np.repeat(np.nan, len(preopt))
feat[ones] = 1
feat[zeros] = 0 
preopt['shock_ecmo_cba'] = feat
preopt.drop(columns=['ecmowhen', 'carshock', 'cathbasassist'], inplace=True)

preopt.loc[preopt.pvd == -9, 'pvd'] = np.nan

preopt.loc[preopt.vdstena == -6, 'vdstena'] = np.nan
preopt.loc[preopt.vdstenm == -6, 'vdstenm'] = np.nan
        
preopt.loc[preopt.vdinsufa == -8, 'vdinsufa'] = np.nan
preopt.loc[preopt.vdinsufa == -6, 'vdinsufa'] = np.nan
preopt.loc[preopt.vdinsufa == 5, 'vdinsufa'] = np.nan
preopt.loc[preopt.vdinsufa == 1, 'vdinsufa'] = 0
preopt.loc[preopt.vdinsufa == 2, 'vdinsufa'] = 0

preopt.loc[preopt.vdinsufm == -8, 'vdinsufm'] = np.nan
preopt.loc[preopt.vdinsufm == -6, 'vdinsufm'] = np.nan
preopt.loc[preopt.vdinsufm == 5, 'vdinsufm'] = np.nan
preopt.loc[preopt.vdinsufm == 1, 'vdinsufm'] = 0
preopt.loc[preopt.vdinsufm == 2, 'vdinsufm'] = 0

preopt.loc[preopt.vdinsuft == -8, 'vdinsuft'] = np.nan
preopt.loc[preopt.vdinsuft == -6, 'vdinsuft'] = np.nan
preopt.loc[preopt.vdinsuft == 5, 'vdinsuft'] = np.nan
preopt.loc[preopt.vdinsuft == 1, 'vdinsuft'] = 0
preopt.loc[preopt.vdinsuft == 2, 'vdinsuft'] = 0
            
#arrhy
recent_cont_pers = preopt.loc[lambda x: (preopt['arrhythafib']==3) &
                              (preopt['arrhythatrfib']==3) |
                              (preopt['arrhythaflutter']==3)].index.values

recent_cont_paro = preopt.loc[lambda x: (preopt['arrhythafib']==2) &
                              (preopt['arrhythatrfib']==3) |
                              (preopt['arrhythaflutter']==3)].index.values

recent_third = preopt[preopt['arrhyththird']==3].index.values

recent_second = preopt.loc[lambda x: (preopt['arrhythsecond']==3) |
                          (preopt['arrhythsss']==3)].index.values

recent_vtach = preopt.loc[lambda x: preopt['arrhythvv']==3].index.values

remote_arr = preopt.loc[lambda x: (preopt['arrhythsss']==2) |
                       (preopt['arrhythatrfib']==2) |
                       (preopt['arrhythvv']==2) |
                       (preopt['arrhythaflutter']==2) |
                       (preopt['arrhyththird']==2) |
                       (preopt['arrhythsecond']==2)].index.values

none = preopt[preopt['arrhythmia']==0].index.values

feat = np.repeat(np.nan, len(preopt))
arr_groups = [recent_cont_pers, recent_cont_paro, recent_third, recent_second,
              recent_vtach, remote_arr, none]
for fi, group in enumerate(arr_groups):
    feat[group] = fi
preopt['arr'] = feat
preopt.drop(columns=['arrhythmia', 'arrhythsss', 'arrhythatrfib', 'arrhythsecond',
                    'arrhythafib', 'arrhythaflutter', 'arrhyththird', 'arrhythvv'], inplace=True)
 

if config['data_procedure'] == 'CABG':
    preopt.drop(columns=['infendty'], inplace=True)
            
preopt.loc[preopt.chrlungd == -9, 'chrlungd'] = np.nan
preopt.loc[preopt.chrlungd == 5, 'chrlungd'] = np.nan

#CVD/CVA/TIA
none = preopt.loc[lambda x: (preopt['cvd']==0) & (preopt['cva']==0) & (preopt['cvdtia']==0)].index.values
cvd_no_cva = preopt.loc[lambda x: (preopt['cvd']==1) & (preopt['cva']==0) & (preopt['cvdtia']==0)].index.values
cvd_tia = preopt.loc[lambda x: (preopt['cvd']==1) & (preopt['cva']==0) & (preopt['cvdtia']==1)].index.values
cvd_cva_remote = preopt.loc[lambda x: (preopt['cvd']==1) & (preopt['cva']==1) & (preopt['cvdtia']==0) & (preopt['cvawhen']==2)].index.values
cvd_cva_recent = preopt.loc[lambda x: (preopt['cvd']==1) & (preopt['cva']==1) & (preopt['cvdtia']==0) & (preopt['cvawhen']==1)].index.values
                     
groups = [none, cvd_no_cva, cvd_tia, cvd_cva_remote, cvd_cva_recent]   
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['cva_cvd_tia'] = feat
preopt.drop(columns=['cva', 'cvawhen', 'cvd', 'cvdtia',], inplace=True)

preopt.loc[preopt.cvdcarsten == -8, 'cvdcarsten'] = np.nan
preopt.loc[preopt.cvdcarsten == 3, 'cvdcarsten'] = 2 # on one side

preopt.loc[preopt.alcohol == -9, 'alcohol'] = np.nan
preopt.loc[preopt.alcohol == 1, 'alcohol'] = 0 

preopt.loc[preopt.ivdrugab == -9, 'ivdrugab'] = np.nan
preopt.loc[preopt.ivdrugab == 4, 'ivdrugab'] = 1
preopt.loc[preopt.ivdrugab == 5, 'ivdrugab'] = 1

preopt.loc[preopt.pneumonia == -9, 'pneumonia'] = np.nan #unknown
preopt.loc[preopt.pneumonia == 3, 'pneumonia'] = 0 #remote pneumonia

preopt.loc[preopt.mediastrad == -9, 'mediastrad'] = np.nan

preopt.loc[preopt.cancer == -9, 'cancer'] = np.nan

# diabetes
none = preopt[preopt['diabetes']==0].index.values
no_ctrl = preopt[preopt['diabctrl']==0].index.values
diet_ctrl = preopt[preopt['diabctrl']==2].index.values
oral_ctrl = preopt[preopt['diabctrl']==3].index.values
insulin_ctrl = preopt[preopt['diabctrl']==4].index.values
                     
groups = [none, no_ctrl, diet_ctrl, oral_ctrl, insulin_ctrl]   
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['diabetes_ctrl'] = feat
preopt.drop(columns=['diabetes', 'diabctrl'], inplace=True)


preopt.loc[preopt.numdisv == 1, 'numdisv'] = 0

if config['data_procedure'] == 'CABG':
    preopt.loc[preopt.miwhen == 4, 'miwhen'] = 3
elif config['data_procedure'] == 'AVR':
    preopt.loc[preopt.miwhen == 2, 'miwhen'] = 1
    preopt.loc[preopt.miwhen == 4, 'miwhen'] = 3
elif config['data_procedure'] == 'AVR_CABG':
    preopt.loc[preopt.miwhen == 2, 'miwhen'] = 1
    preopt.loc[preopt.miwhen == 3, 'miwhen'] = 1
    preopt.loc[preopt.miwhen == 4, 'miwhen'] = 1
    
preopt.loc[preopt.cardsymptimeofadm == 2, 'cardsymptimeofadm'] = 1
preopt.loc[preopt.cardsymptimeofadm == 7, 'cardsymptimeofadm'] = np.nan
preopt.loc[preopt.cardsymptimeofadm == 8, 'cardsymptimeofadm'] = np.nan

white = preopt[preopt['racecaucasian']==1].index.values
black = preopt[preopt['raceblack']==1].index.values 
asian = preopt[preopt['raceasian']==1].index.values 
native = preopt[preopt['racenativeam']==1].index.values 
groups = [white, black, asian, native]   
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['race'] = feat
preopt.drop(columns=['racecaucasian', 'raceblack', 'raceasian', 'racenativeam'], inplace=True)    

preopt.loc[preopt.medacei48 == -9, 'medacei48'] = np.nan
preopt.loc[preopt.medacei48 == 3, 'medacei48'] = 0
preopt.loc[preopt.status == 1, 'medacei48'] = 0 # status=elective

none = preopt.loc[lambda x: (preopt['chf']==0) & (preopt['priorhf']==0)].index.values
hf_recent = preopt[preopt['chf']==1].index.values
hf_remote = preopt[preopt['priorhf']==1].index.values
groups = [none, hf_remote, hf_recent]   
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['hf'] = feat
preopt.drop(columns=['chf', 'priorhf'], inplace=True)

preopt.loc[preopt.tobaccouse == 3, 'tobaccouse'] = np.nan
preopt.loc[preopt.tobaccouse == 4, 'tobaccouse'] = np.nan
preopt.loc[preopt.tobaccouse == 5, 'tobaccouse'] = np.nan
preopt.loc[preopt.tobaccouse == 6, 'tobaccouse'] = np.nan

preopt.loc[preopt.fhcad == -9, 'fhcad'] = np.nan

preopt.loc[preopt.hmo2 == -9, 'hmo2'] = np.nan
preopt.loc[preopt.hmo2 == 3, 'hmo2'] = 1
preopt.loc[preopt.hmo2 == 4, 'hmo2'] = 1

preopt.loc[preopt.slpapn == -9, 'slpapn'] = np.nan

preopt.loc[preopt.liverdis == -9, 'liverdis'] = np.nan

preopt.loc[preopt.unrespstat == -9, 'unrespstat'] = np.nan

preopt.loc[preopt.syncope == -9, 'syncope'] = np.nan

preopt.loc[preopt.incidenc == 2, 'incidenc'] = 1
preopt.loc[preopt.incidenc == 5, 'incidenc'] = 4
preopt.loc[preopt.incidenc == 6, 'incidenc'] = np.nan

#PCI
none = preopt.loc[lambda x: preopt['pocpci']==0].index.values
pci_remote = preopt.loc[lambda x: (preopt['pocpci']==1) & (preopt['pocpciwhen']==0)].index.values

if config['data_procedure']=='CABG':
    pci_recent_6g = preopt.loc[lambda x: (preopt['pocpci']==1) & (preopt['pocpciin']==2) & (preopt['pocpciwhen']==1) | (preopt['pocpciwhen']==2)].index.values
    pci_recent_6l = preopt.loc[lambda x: (preopt['pocpci']==1) & (preopt['pocpciin']==1) & (preopt['pocpciwhen']==1) | (preopt['pocpciwhen']==2)].index.values
    groups = [none, pci_remote, pci_recent_6g, pci_recent_6l] 

elif config['data_procedure']!='CABG': 
    pci_recent = preopt.loc[lambda x: (preopt['pocpci']==1) & (preopt['pocpciwhen']==1) | (preopt['pocpciwhen']==2)].index.values
    groups = [none, pci_remote, pci_recent] 
  
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['pci'] = feat
preopt.drop(columns=['pocpci', 'pocpciin', 'pocpciwhen'], inplace=True)


#payor
medicareg = preopt.loc[lambda x: (preopt['AGE']>=65) & (preopt['payorprim']==2)].index.values
hmog = preopt.loc[lambda x: (preopt['AGE']>=65) & (preopt['payorprim']==9) | (preopt['payorprim']==10)].index.values
otherg = preopt.loc[lambda x: (preopt['AGE']>=65) & (preopt['payorprim']!=2) & 
                        (preopt['payorprim']!=9) & (preopt['payorprim']!=10)].index.values
medicarel = preopt.loc[lambda x: (preopt['AGE']<65) & (preopt['payorprim']==2)].index.values
medicaidl = preopt.loc[lambda x: (preopt['AGE']<65) & (preopt['payorprim']==3)].index.values
hmol = preopt.loc[lambda x: (preopt['AGE']<65) & (preopt['payorprim']==9) | (preopt['payorprim']==10)].index.values
selfl = preopt.loc[lambda x: (preopt['AGE']<65) & (preopt['payorprim']==1)].index.values
otherl = preopt.loc[lambda x: (preopt['AGE']<65) & (preopt['payorprim']!=1) & 
                        (preopt['payorprim']!=2) & (preopt['payorprim']!=3) & 
                        (preopt['payorprim']!=9) & (preopt['payorprim']!=10)].index.values

groups = [medicareg, hmog, otherg, medicarel, medicaidl, hmol, selfl, otherl]
feat = np.repeat(np.nan, len(preopt))
for fi, group in enumerate(groups):
    feat[group] = fi
preopt['payer'] = feat
preopt.drop(columns=['payorprim'], inplace=True)


for feat in config['allphase'].values():
    preopt.rename(columns= {feat: feat+'_preopt'}, inplace=True)
    
ohot_feats = pd.DataFrame()
for feat in preopt.columns:
    if sum(preopt[feat].isnull()) > len(preopt[feat])*0.99:
        preopt.drop(columns=feat, inplace=True)
        continue
        
    if len(preopt[feat].unique()) < 9:
        new_feats = _ohe_feats(preopt, feat, preopt.index.name)
        ohot_feats = pd.concat([ohot_feats, new_feats], axis=1)
        preopt.drop(columns=feat, inplace=True)

preopt = pd.concat([preopt, ohot_feats], axis=1)

preopt.to_csv('{}_Cohort_Preopt_Data.csv'.format(config['data_procedure']), index=False) 
