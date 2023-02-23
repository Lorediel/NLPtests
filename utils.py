from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import Dataset, Subset
import pandas as pd
from NLPtests.FakeNewsDataset import FakeNewsDataset



def splitDataset(dataset): 
    # 80% train, 20% validation
    validation_split = 0.2
    random_seed = 64

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)
    
    return train_dataset, validation_dataset


def splitTrainTestVal(filepath, delete_date = False):
    data_files = {"train": filepath}
    dataset_dict = load_dataset("csv", data_files=filepath, delimiter="\t")
    if (delete_date):
        dataset_dict = dataset_dict.remove_columns(["Date"])
    dataset = dataset_dict["train"]
    dataset = dataset.shuffle(seed = 64)
    train_testvalid = dataset.train_test_split(test_size=0.2)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'valid': train_testvalid['test']})
    return train_test_valid_dataset

def splitTrainTestVal2(filepath, dataset, delete_date = False):
    indexes = take_per_type_label_indexes(dataset)
    train_indexes = []
    validation_indexes = []
    for k,v in indexes.items():
        t_i, v_i = get_train_val_indexes(v)
        train_indexes += t_i
        validation_indexes += v_i
    
    data_files = {"train": filepath}
    dataset_dict = load_dataset("csv", data_files=filepath, delimiter="\t")
    if (delete_date):
        dataset_dict = dataset_dict.remove_columns(["Date"])
    dataset = dataset_dict["train"]
    # split the dataset based on train_indexes and validation
    train_ds = dataset.select(train_indexes)
    validation_ds = dataset.select(validation_indexes)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
    'train': train_ds,
    'valid': validation_ds})
    return train_test_valid_dataset



def build_dataloaders(tokenized_ds, data_collator, batch_size = 8):
        train_dataloader = DataLoader(
        tokenized_ds["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_ds["valid"], batch_size=batch_size, collate_fn=data_collator
        )

        return train_dataloader, eval_dataloader

def compute_precision(preds, ground_truth, average = 'macro'):
    return precision_score(ground_truth, preds, average=average, zero_division=1)

def compute_recall(preds, ground_truth, average = 'macro'):
    return recall_score(ground_truth, preds, average=average, zero_division=1)

def compute_f1(preds, ground_truth, average = 'macro'):
    return f1_score(ground_truth, preds, average=average, zero_division=1)

def compute_f1_None(preds, ground_truth):
    return f1_score(ground_truth, preds, average=None, zero_division=1)

def compute_accuracy(preds, ground_truth):
    return accuracy_score(ground_truth, preds)

def compute_metrics(preds, ground_truth):
    return {
        "accuracy": round(compute_accuracy(preds, ground_truth),3),
        "precision": round(compute_precision(preds, ground_truth),3),
        "recall": round(compute_recall(preds, ground_truth),3),
        "f1": round(compute_f1(preds, ground_truth),3),
        "f1_none": [round(n, 3) for n in compute_f1_None(preds, ground_truth)],
    }


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


if __name__ == '__main__':
    ds = FakeNewsDataset("/Users/lorenzodamico/Documents/Uni/tesi/NLPtests/MULTI-Fake-Detective_Task1_Data.tsv", "/Users/lorenzodamico/Documents/Uni/tesi/content/Media")
    
    train, val = stratifiedSplit(ds)
    print(train[0])
    print(val[0])
    print(train[54])
    print(val[54])

    ds = splitTrainTestVal2("/Users/lorenzodamico/Documents/Uni/tesi/NLPtests/MULTI-Fake-Detective_Task1_Data.tsv", ds)
    print(ds["train"][0])
    print(ds["valid"][0])
    print(ds["train"][54])
    print(ds["valid"][54])