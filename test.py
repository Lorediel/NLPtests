from resnet import ResnetModel
from datasets import load_dataset
from utils import splitTrainTestVal

resnet = ResnetModel()
datasets = splitTrainTestVal("/content/NLPtests/MULTI-Fake-Detective_Task1_Data.tsv", delete_date = True)
resnet.test(datasets)