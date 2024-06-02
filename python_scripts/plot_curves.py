import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, classification_report
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import os
def plot_roc_test(classifiers, test, features, label, path):

    for key, val in classifiers.items(): #for each type of classifier do an ensamble of the respective trained models

        classifier_name= val[0].__class__.__name__

        X_test = test[features].values
        y_test = test[[label]].values

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_classes = 2

        #ensamble: take the mean of prediction and then the argmax on the probabilities of class 0 and 1

        y_score = np.array([c.predict_proba(X_test) for c in val])
        y_score_means = np.mean(y_score, axis=0)
        y_pred = np.argmax(y_score_means, axis=1)

        #plot conf matrix and ROC curves

        conf_mtx = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mtx,
                display_labels = val[0].classes_)
        plt.title("Confusion Matrix on Test Set with {} model".format(classifier_name))

        disp.plot()
        plt.savefig(os.path.join(path, classifier_name + "_ConfMatrix_test.png"))
        plt.close()
       #plt.show()

        for i in range(n_classes):
            if i > 0:
                fpr[i], tpr[i], _ = roc_curve(y_test[:, 0], y_score_means[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

                plt.plot(
                    fpr[i],
                    tpr[i],
                    lw=2,
                    label="ROC curve for class {} (area = {:.2f})".format(i, roc_auc[i]),
                )

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic for Each Class with {} model".format(classifier_name))
        plt.legend(loc="lower right")
        #plt.show()

        plt.savefig(os.path.join(path, classifier_name+"_crossVal_test.png"))
        plt.close()

        #print evaluation metrics
        overall_prec, overall_rec, overall_fscore, _ = precision_recall_fscore_support(y_test[:, 0], y_pred,  average='weighted')
        class_report = classification_report(y_test[:, 0], y_pred)
        print(f"Precision, Recall and F-Score for {classifier_name} model: {overall_prec} - {overall_rec} - {overall_fscore}")
        print(class_report)

