from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

def splitTrainTestVal(filepath):
    data_files = {"train": filepath}
    dataset_dict = load_dataset("csv", data_files=filepath, delimiter="\t")
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



if __name__ == "__main__":
    d = splitTrainTestVal("./MULTI-Fake-Detective_Task1_Data.tsv")
    print(d['train'].set_format("torch", columns=["input_ids", "attention_mask", "labels"]))