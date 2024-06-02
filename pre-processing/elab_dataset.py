
from utils.utils import *
from utils.statistical_analysis import statistical_test


if __name__ == '__main__':
    path_to_data = r'dataset/SEVERE OSA.xlsx' #load patient data
    data = read_data(path_to_data)

    data_cpap_dup = pd.read_excel(r"dataset/DatabasePNEUMO.xlsx") #load additional information about CPAP treatment
    data_cpap = data_cpap_dup.drop_duplicates(subset=['num_cartella'], keep='first')

    data = data.rename(columns={"N CC": "num_cartella"})
    data2 = data.drop_duplicates(subset=['num_cartella'], keep='first')

    new_data = pd.merge(data2, data_cpap, on='num_cartella', how='inner')  #patient data joined with CPAP information



    data_prep = preprocess(data) #dropping non relevant features and categorization
    data_cpap = preprocessing_CPAP(data_prep) #preprocessing CPAP data
    # data_cpap.to_csv(r'/Volumes/ExtremeSSD/pycharmProg/surv_OSA/dataset/full_data_.csv', index=False) #save full data
    train, test = train_test_time_split(data_cpap)

    data_elab = statistical_test(train) #statistical test and feature selection based on correlation
    features = train.columns
    test = test[features]

    #train.to_csv(r"dataset/train_data.csv",index=False)
    #test.to_csv(r"dataset/test_data.csv",index=False)
