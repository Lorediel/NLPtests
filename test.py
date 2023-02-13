from utils import splitTrainTestVal
from bertModel import BertModel

# Load the dataset
datasets = splitTrainTestVal("./MULTI-Fake-Detective_Task1_Data.tsv")

# Initialize the model
model = BertModel(datasets)