from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import qa_data

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from utils.dataset import DataSet
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize

from pdb import set_trace as t

def load_and_preprocess_data(debug = False):
	dataset = DataSet() 
	stances = dataset.stances
	articles = dataset.articles
	if debug:
		allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance']) for s in stances[:200]]
#		allData = [(word_tokenize(s['Headline'].decode('utf-8')), word_tokenize(articles[s['Body ID']].decode('utf-8')), \
#			s['Stance']) for s in stances[:200]]
	else:
		allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance']) for s in stances]
#		allData = [(word_tokenize(s['Headline'].decode('utf-8')), word_tokenize(articles[s['Body ID']].decode('utf-8')), \
#			s['Stance']) for s in stances]
	
	return allData

if __name__ == "__main__":

	# get all the articles
	allData = load_and_preprocess_data(False)

	# create the dictionary
	vocab = {}
	for entry in allData:
		for word in entry[0] + entry[1]:
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1

	qa_data.create_vocabulary_withVocab(vocabulary_path = './myVocab.txt', vocab = vocab)