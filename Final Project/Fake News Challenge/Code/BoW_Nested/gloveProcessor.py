
import argparse
import logging
import sys
import time
from datetime import datetime


from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from utils.dataset import DataSet
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize

from pdb import set_trace as t

# open the read and write files
inFile = open('data/glove.6B.50d.txt', 'r')
outFile = open('data/6bVocab.txt', 'w')


# get list of words in vocab
vocabList = []
for line in inFile:
	vocabList.append(line.split()[0])

for v in vocabList:
	outFile.write('%s\n' % v)
