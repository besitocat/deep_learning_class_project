import utils
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compute_accuracy(preds, targets):
    pass


def compute_metrics(labels, targets):
    conf_matrix = confusion_matrix(targets, labels)
    print "\nConfusion matrix:\n", conf_matrix
    target_names = ['rating 1', 'rating 2', 'rating 3', 'rating 4', 'rating 5']
    print(classification_report(targets, labels, target_names=target_names))
    accuracy = accuracy_score(targets, labels)
    print("Accuracy %.3f"%accuracy)
#    auc_score = auc(targets, labels)
#    print("AUC %.3f" % auc_score)
    df_cm = pd.DataFrame(conf_matrix, index=target_names,
                  columns = target_names)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

def probs_to_labels(probs):
#    classes = np.array([0, 1, 2, 3, 4])
#    return list(sorted(zip(classes_, probs), key=lambda k: -k[1]))[1]
#    return probs.index(min(probs))
   return np.argmax(probs,axis=None)


if __name__=="__main__":
    
#    probs_filepath = "experiments/results/ffn_predictions.txt"
#    targets_filepath = "experiments/data/y_val_noseq.pickle"
#
#    targets = utils.load_pickle_file(targets_filepath)
#    predictions = utils.load_predictions(probs_filepath)
#    targets = probs_to_labels(targets)
#    labels = probs_to_labels(predictions)
#    compute_metrics(labels, targets)
    
    probs = [[0.4, 0.5, 0.2, 0.3, 0.1], [0.9, 0.3, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.9, 0.3], [0.4, 0.3, 0.3, 0.8, 0.9], [0.2, 0.3, 0.4, 0.3, 0.4]]
    labels = list()
    for p in probs:
        print p
        labels.append(probs_to_labels(p))


    print labels
    targets = [1, 0, 2, 3, 4]
    compute_metrics(labels, targets)
