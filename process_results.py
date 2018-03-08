import utils
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import numpy as np


def compute_accuracy(preds, targets):
    pass


def compute_metrics(labels, targets):
    conf_matrix = confusion_matrix(targets, labels)
    print "\nConfusion matrix:\n", conf_matrix
    target_names = ['class 0 (not sarcasm)', 'class 1 (sarcasm)']
    print(classification_report(targets, labels, target_names=target_names))
    accuracy = accuracy_score(targets, labels)
    print("Accuracy %.3f"%accuracy)
    auc_score = auc(targets, labels)
    print("AUC %.3f" % auc_score)

def probs_to_labels(probs):
    return np.argmax(probs,axis=1)


if __name__=="__main__":
    probs_filepath = "experiments/results/ffn_predictions.txt"
    targets_filepath = "experiments/data/y_val_noseq.pickle"

    targets = utils.load_pickle_file(targets_filepath)
    predictions = utils.load_predictions(probs_filepath)
    targets = probs_to_labels(targets)
    labels = probs_to_labels(predictions)
    compute_metrics(labels, targets)
