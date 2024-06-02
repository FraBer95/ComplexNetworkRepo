import pandas as pd
from train import training_models
from plot_curves import plot_roc_test


if __name__ == '__main__':

    #read data
    train = pd.read_csv("/Volumes/ExtremeSSD/pycharmProg/surv_OSA/dataset/data_CPAP_train_new1.csv")
    test = pd.read_csv("/Volumes/ExtremeSSD/pycharmProg/surv_OSA/dataset/data_CPAP_test_new1.csv")

    X_train = train.drop(columns=['Status', 'Durata follow-up da dimissione']) #drop event time and target
    y_train = train['Status'] #target variable
    classifiers = training_models(X_train, y_train, save_path='./logs') #training models
    test = test.drop(columns=['Durata follow-up da dimissione'])
    plot_roc_test(classifiers, test, X_train.columns, 'Status', path='./logs') #validation on test set

