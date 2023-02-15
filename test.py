from resnet import ResnetModel
from datasets import load_dataset

resnet = ResnetModel()
datasets = load_dataset("csv", data_files="./MULTI-Fake-Detective_Task1_Data.tsv", delimiter="\t")
resnet.test(datasets)