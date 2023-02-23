from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import Dataset, Subset

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
    # 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.2)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'valid': train_testvalid['test']})
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

def compute_accuracy(preds, ground_truth):
    return accuracy_score(ground_truth, preds)

def compute_metrics(preds, ground_truth):
    return {
        "accuracy": round(compute_accuracy(preds, ground_truth),3),
        "precision": round(compute_precision(preds, ground_truth),3),
        "recall": round(compute_recall(preds, ground_truth),3),
        "f1": round(compute_f1(preds, ground_truth),3),
    }
