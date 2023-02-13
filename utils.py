from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

def splitTrainTestVal(filepath):
    data_files = {"train": filepath}
    dataset_dict = load_dataset("csv", data_files=filepath, delimiter="\t")
    dataset = dataset_dict["train"]
    dataset = dataset.shuffle(seed = 64)
    # 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
    return train_test_valid_dataset
    
def build_dataloaders(tokenized_ds, data_collator):
        train_dataloader = DataLoader(
        tokenized_ds["train"], shuffle=True, batch_size=8, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_ds["valid"], batch_size=8, collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            tokenized_ds["test"], batch_size=8, collate_fn=data_collator
        )

if __name__ == "__main__":
    d = splitTrainTestVal("./MULTI-Fake-Detective_Task1_Data.tsv")
    print(d['train'].set_format("torch", columns=["input_ids", "attention_mask", "labels"]))