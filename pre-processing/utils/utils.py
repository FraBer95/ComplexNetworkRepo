import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn. model_selection import train_test_split
def read_data(path):
    data = pd.read_excel(path)
    data.rename(columns={'Status in vita (0=vivo; 1=morto)': 'Status',
                         'Stato civile (0=coniugato; 1=non coniugato)': 'Status Civile', 'Sesso (0=F;1=M)': 'Sesso'
        , 'Normal weight (18.5-24.9)': 'Normal weight', 'Overweight (25-<30)': 'Overweight',
                         'Obesity class 1 (30 to <35)': 'Obesity class I'
        , 'Obesity class II (35 to <40)': 'Obesity class II', 'MORBID OBESITY (≥ 40)': 'Morbid Obesity'
        , 'ODI (NA=not available)': 'ODI'
        ,
                         'Cardiovascular disease (CAD/HF/AF/Previous CVE/Idiopathic dilated cardiomyopathy/Valvular heart disease)': 'Cardiovascular disease'
        , 'Renal dysfunction (0=no; 1=sì)': 'Renal dysfunction',
                         'Colesterolo categorie (1=<200; 2=200 to 239; 3=≥240)': 'Colesterolo categorie'}, inplace=True)
    return data


def preprocess(data):
    drop_list = ["Anemia", "FE", "ODI", "Data Ricovero", "Data Dimissione", "Data Follow-up", 'I.D.']

    print("Dropping: {}".format(drop_list))

    data.drop(columns=drop_list, axis=1, inplace=True)

    for col in data.columns:
        if col in ["Normal weight", "Overweight", "Obesity class I", "Obesity class II", "Morbid Obesity"]:
            data[col].fillna(0, inplace=True)

    data['prof_descr'].fillna('ALTRO                                        ', inplace=True)

    data["GOLD"].fillna("GOLD 0", inplace=True)

    lavoro_leggero = ['PENSIONATO (che ha svolto lavoro retribuito) ',
                      'CASALINGA (non ha mai svolto lavoro retr.)   ',
                      'CASALINGA (che ha svolto lavoro retribuito)  ',
                      'SCOLARO, STUDENTE, BAMBINO                   '
                      ]
    lavoro_moderato = ['IMPRENDITORE, LIBERO PROFESSIONISTA          ',
                       'IMPIEGATO, INSEGNANTE                        ',
                       'DIRIGENTE                                    ',
                       'COADIUVANTE IN AZIENDE A CONDUZIONE FAMILIARE',
                       'LAVORANTE A DOMICILIO                        '
                       ]
    invalido = ['INVALIDO,INABILE (mai svolto lavoro retr.)   ',
                'INVALIDO,INABILE (anche se pensionato)       '
                ]
    lavoro_pesante = ['OPERAIO, ALTRO LAVORATORE DIPENDENTE         ',
                      'ARTIGIANO, ALTRO LAVORATORE IN PROPRIO       ']
    altro = ['ALTRO                                        ',
             'IN CERCA DI PRIMA OCCUPAZIONE                ',
             'DISOCCUPATO (attualmente)                    ']

    data['prof_descr'].loc[(data['prof_descr'].isin(lavoro_leggero))] = 1
    data['prof_descr'].loc[(data['prof_descr'].isin(lavoro_moderato))] = 2
    data['prof_descr'].loc[(data['prof_descr'].isin(invalido))] = 3
    data['prof_descr'].loc[(data['prof_descr'].isin(lavoro_pesante))] = 4
    data['prof_descr'].loc[(data['prof_descr'].isin(altro))] = 5
    data['prof_descr'].astype(np.float64)
    #
    data.loc[(data['Age'] <= 17), 'Age'] = 0
    data.loc[(data['Age'] > 17) & (data['Age'] <= 35), 'Age'] = 1
    data.loc[(data['Age'] > 35) & (data['Age'] <= 53), 'Age'] = 2
    data.loc[(data['Age'] > 53) & (data['Age'] <= 71), 'Age'] = 3
    data.loc[(data['Age'] > 71), 'Age'] = 4

    print(data['prof_descr'].unique())

    # data.dropna(inplace = True)

    data["GOLD"].replace("GOLD 0", 0.0, inplace=True)
    data["GOLD"].replace("GOLD 1", 1.0, inplace=True)
    data["GOLD"].replace("GOLD 2", 2.0, inplace=True)
    data["GOLD"].replace("GOLD 3", 3.0, inplace=True)
    data["GOLD"].replace("GOLD 4", 4.0, inplace=True)

    # Computing Anemia from Hemoglobin
    data['Anemia'] = 0.0
    mask_men = (data['Sesso'] == 1) & (data['Emoglobina'] <= 13)
    mask_wom = (data['Sesso'] == 0) & (data['Emoglobina'] <= 12)

    data.loc[mask_men | mask_wom, 'Anemia'] = 1.0
    data.drop(columns=["Emoglobina", "Ematocrito"], axis=1, inplace=True)

    # Creo delle categorie basate sul valore del GFR
    data['GFR Categories'] = pd.cut(data['GFR'],
                                    bins=[-float('inf'), 29, 44, 59, float('inf')],
                                    labels=['3', '2', '1', '0'],
                                    right=False)
    data.drop(columns=["GFR"], axis=1, inplace=True)

    # BMI categories
    weight_class = ['Normal weight', 'Overweight', 'Obesity class I', 'Obesity class II', 'Morbid Obesity']
    data.drop(columns=weight_class, axis=1, inplace=True)
    data['BMI Categories'] = pd.cut(data['BMI'],
                                    bins=[18.0, 25, 30, 35, 39.9, float('inf')],
                                    labels=['0', '1', '2', '3', '4'],
                                    right=False)
    data.drop(columns=["GFR"], axis=1, inplace=True)
    data['BMI Categories'] = data['BMI Categories'].astype(float)

    return data


def preprocessing_CPAP(data):
    #data = pd.read_csv(path,sep=';')

    drop_list = ["id_cod"]
    print("Dropping: {}".format(drop_list))
    data.drop(columns=drop_list, axis=1, inplace=True)
    data['CPAP_bin'] = data['CPAP'].notna().astype(int)
    data['CPAP'] = pd.to_datetime(data['CPAP'])
    data['ricovero_data'] = pd.to_datetime(data['ricovero_data'])
    data['Years_of_CPAP'] = ((data['CPAP'] - data['ricovero_data']).dt.total_seconds() / (365.25 * 24 * 3600)).fillna(0)

    data['CPAP_0_5'] = data['Years_of_CPAP'].apply(lambda x: 1 if 0 < x <= 5 else 0)
    data['CPAP_5_10'] = data['Years_of_CPAP'].apply(lambda x: 1 if 5 < x <= 10 else 0)
    data['CPAP>10'] = data['Years_of_CPAP'].apply(lambda x: 1 if x > 10 else 0)

    data['CPAP_cat'] = data.apply(lambda row: 0 if row['CPAP'] == 0 else
    1 if row['CPAP_0_5'] == 1 else
    2 if row['CPAP_5_10'] == 1 else
    3 if row['CPAP>10'] == 1 else 0, axis=1)

    data = data[(data['CPAP'].isnull()) | (data['CPAP'].dt.year <= 2020)]
    drop_list = [ "ricovero_data", "CPAP", "sesso", "nascita_data", "ricovero_rep_cod", "num_cartella"]
    data_processed = data
    data_processed.drop(columns=drop_list, axis=1, inplace=True)
    data_processed.rename(columns={'CPAP_bin': 'CPAP'}, inplace=True)
    print("Dropping: {}".format(drop_list))



    return data


def train_test_time_split(data):
    for state in tqdm(range(0, 1000)):
        # divide data in training and test
        train, test = train_test_split(data, random_state=state, test_size=0.30, shuffle=True,
                                       stratify=data[['Status']])
        time = 'Durata follow-up da dimissione'
        train[time] = train[time]
        test[time] = test[time]

        # range values for observation time in test
        durata_follow_up_min_test = test['Durata follow-up da dimissione'].min()
        durata_follow_up_max_test = test['Durata follow-up da dimissione'].max()

        # range values for observation time in train
        durata_follow_up_min_train = train['Durata follow-up da dimissione'].min()
        durata_follow_up_max_train = train['Durata follow-up da dimissione'].max()

        # check if the test range is contained in training
        if durata_follow_up_min_test >= durata_follow_up_min_train and durata_follow_up_max_test <= durata_follow_up_max_train:
            print(f"Random state {state}")
            # train.to_csv('./dataset/data_CPAP_train.csv', index=False)
            # test.to_csv('./dataset/data_CPAP_test.csv', index=False)
            return train, test