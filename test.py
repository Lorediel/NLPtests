from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset
from resnet import ResnetModel
from utils import splitTrainTestVal

resnet = ResnetModel()
datasets = splitTrainTestVal("/content/NLPtests/MULTI-Fake-Detective_Task1_Data.tsv", delete_date = True)
resnet.test(datasets, "/content/content/Media")
