from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from torch.utils.data import Dataset, Subset
import pandas as pd
#from FakeNewsDataset import FakeNewsDataset
import collections
import random
import matplotlib.pyplot as plt
def compute_precision(preds, ground_truth, average = 'macro'):
    return precision_score(ground_truth, preds, average=average, zero_division=1)

def compute_recall(preds, ground_truth, average = 'macro'):
    return recall_score(ground_truth, preds, average=average, zero_division=1)

def compute_f1(preds, ground_truth, average = 'macro'):
    return f1_score(ground_truth, preds, average=average, zero_division=1)

def compute_f1_weighted(preds, ground_truth):
    return f1_score(ground_truth, preds, average='weighted', zero_division=1)

def compute_f1_None(preds, ground_truth):
    return f1_score(ground_truth, preds, average=None, zero_division=1)

def compute_accuracy(preds, ground_truth):
    return accuracy_score(ground_truth, preds)

def compute_metrics(preds, ground_truth):
    metrics =  {
        "accuracy": round(compute_accuracy(preds, ground_truth),3),
        "precision": round(compute_precision(preds, ground_truth),3),
        "recall": round(compute_recall(preds, ground_truth),3),
        "f1": round(compute_f1(preds, ground_truth),3),
        "f1_weighted": round(compute_f1_weighted(preds, ground_truth),3),
        "f1_none": [round(n, 3) for n in compute_f1_None(preds, ground_truth)],
    }
    return metrics

def format_metrics(metrics):
    formatted = ""
    for k,v in metrics.items():
        formatted += str(v) + "\t "
    formatted = formatted[:-1]
    return formatted

def take_per_type_label_indexes(dataset):
    indexes = {
        "tweet_0": [],
        "tweet_1": [],
        "tweet_2": [],
        "tweet_3": [],
        "article_0": [],
        "article_1": [],
        "article_2": [],
        "article_3": []
    }
    for i in range(len(dataset)):
        x = dataset[i]
        if (x["type"] == "tweet"):
            if (x["label"] == 0):
                indexes["tweet_0"].append(i)
            elif (x["label"] == 1):
                indexes["tweet_1"].append(i)
            elif (x["label"] == 2):
                indexes["tweet_2"].append(i)
            elif (x["label"] == 3):
                indexes["tweet_3"].append(i)
        else:
            if (x["label"] == 0):
                indexes["article_0"].append(i)
            elif (x["label"] == 1):
                indexes["article_1"].append(i)
            elif (x["label"] == 2):
                indexes["article_2"].append(i)
            elif (x["label"] == 3):
                indexes["article_3"].append(i)
    return indexes

def stratifiedSplit(dataset):
    indexes = take_per_type_label_indexes(dataset)
    train_indexes = []
    validation_indexes = []
    for k,v in indexes.items():
        t_i, v_i = get_train_val_indexes(v)
        train_indexes += t_i
        validation_indexes += v_i


    random.Random(64).shuffle(train_indexes)
    random.Random(64).shuffle(validation_indexes)

    train_dataset = Subset(dataset, train_indexes)
    validation_dataset = Subset(dataset, validation_indexes)
    return train_dataset, validation_dataset

def get_train_val_indexes(indexes):
    random_seed = 64
    np.random.seed(random_seed)
    np.random.shuffle(indexes)
    split = int(np.ceil(0.2 * len(indexes)))
    train_indexes = indexes[split:]
    validation_indexes = indexes[:split]
    return train_indexes, validation_indexes

def get_only_type(dataset, type):
    indexes = []
    for i in range(len(dataset)):

        x = dataset[i]
        if (x["type"] == type):
            indexes.append(i)
    return Subset(dataset, indexes)

def stratified_by_label(dataset):
    indexes = {
        "0": [],
        "1": [],
        "2": [],
        "3": []
    }
    for i in range(len(dataset)):
        x = dataset[i]
        if (x["label"] == 0):
            indexes["0"].append(i)
        elif (x["label"] == 1):
            indexes["1"].append(i)
        elif (x["label"] == 2):
            indexes["2"].append(i)
        elif (x["label"] == 3):
            indexes["3"].append(i)
    train_indexes = []
    validation_indexes = []
    for k,v in indexes.items():
        t_i, v_i = get_train_val_indexes(v)
        train_indexes += t_i
        validation_indexes += v_i
    random.Random(64).shuffle(train_indexes)
    random.Random(64).shuffle(validation_indexes)
    train_dataset = Subset(dataset, train_indexes)
    validation_dataset = Subset(dataset, validation_indexes)
    return train_dataset, validation_dataset
    
def cf1(percents, n_samples):
    f1 = 0
    for i in range(len(percents)):
        f1 += percents[i] * n_samples[i]
    return f1 / sum(n_samples)

def get_confusion_matrix(preds, ground_truth):
    return confusion_matrix(ground_truth, preds)

def display_confusion_matrix(ground_truth, preds):
    cm = get_confusion_matrix(preds, ground_truth)
    ConfusionMatrixDisplay(cm, display_labels=["Certainly Fake", "Probably Fake", "Probably Real", "Certainly Real"]).plot()
    plt.show()

if __name__ == '__main__':
    # make a random list of predictions and ground truth of length 1000
    preds = [random.randint(0,3) for i in range(1000)]
    ground_truth = [0] * 100 + [1] * 300 + [2] * 350 + [3] * 250
    random.Random().shuffle(ground_truth)

    print(compute_metrics(preds, ground_truth)) 
    print(compute_metrics(ground_truth, preds))