#from msilib import sequence
from torchtext.legacy import data
from torchtext.legacy import datasets

TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.Field(sequential = False)

train, test = datasets.IMDB.splits(TEXT, LABEL)
