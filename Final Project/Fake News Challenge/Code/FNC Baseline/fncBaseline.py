from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import csv 

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from utils.dataset import DataSet
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from fnc_utils.dataset import DataSet
from fnc_utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_utils.score import report_score, LABELS, LABELS_RELATED, score_submission
from sklearn.ensemble import GradientBoostingClassifier
from pdb import set_trace as t
from sklearn.metrics import confusion_matrix


from pdb import set_trace as t

class Config:

	headline_hidden_size = 200
	article_hidden_size = 200

	batch_size = 32
	embed_size = 200 # try a larger embedding
	max_length_titles = 40
	max_length_articles = 40 #changed from 700
	num_classes = 4
	epochs = 10
	lr = 0.001
	dropout = 1.0
	final_state_size = 50

# feature generation for binary model
def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

def load_and_preprocess_data(debug = False):
    dataset = DataSet() 
    stances = dataset.stances
    articles = dataset.articles
    if debug:
        allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance'], s['Body ID']) for s in stances[:200]]
    else:
        allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance'], s['Body ID']) for s in stances]
     
    # choose specific articles to comprise the training set
    articleList = articles.keys()
    np.random.seed(2017)
    np.random.shuffle(articleList)
 
    trainArticleIndices = articleList[:int(len(articleList)*4/5)]
    devArticleIndices = articleList[int(len(articleList)*4/5):]
 
    train = [v[:3] for v in allData if v[3] in trainArticleIndices]
    dev = [v[:3] for v in allData if v[3] in devArticleIndices]
    allData = [v[:3] for v in allData]
 
 
    helper = ModelHelper.build(allData)
 
    # now process all the input data.
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)
 
    return helper, train_data, dev_data, train, dev, trainArticleIndices, devArticleIndices, dataset


def generate_batches(data):

	# shuffle the data
	np.random.seed(2017)
	np.random.shuffle(data)

	# create batches
	batchNum = int(np.ceil(len(data)/float(Config.batch_size)))
	batches = []
	for i in range(batchNum):
		base = i*Config.batch_size
		batches.append(data[base:(min(base + Config.batch_size, len(data)))])
	return batches


if __name__ == "__main__":

	######################################
	##           prompt users           ##	
	######################################

	print 'What should the accuracy file name be?'
	accuracyFileName = raw_input()
	print 'What should the confusion matrix file name be?'
	cmFileName = raw_input()

	######################################
	##           get the data           ##	
	######################################

	# load in the data
	debug = False  
	if len(sys.argv) > 2 and sys.argv[2] == "debug":
		debug = True
	helper, train, dev, train_raw, dev_raw, trainArticleIndices, devArticleIndices, d = load_and_preprocess_data(debug)
	pretrained_embeddings = load_embeddings(helper, vocabPath = "../Baselines_w_RELU/data/gloveVocab.txt", 
		vectorPath = "../Baselines_w_RELU/data/glove.6B.200d.txt", wordFirst = True, embed_size = 200)
	Config.embed_size = pretrained_embeddings.shape[1]

	# for later
	neverOpened_gold = True 
	neverOpened_test = True 


	######################################
	##           train model            ##	
	######################################	


	# train model on related vs. unrelated
	train_stances, devStances = get_stances_for_folds(d, [trainArticleIndices], devArticleIndices)
	train_stances = train_stances[0]

	# load/precompute all features now
	X_train, Y_train = generate_features(train_stances, d, "train")
	X_dev, Y_dev = generate_features(devStances, d, "dev")

	# build the model 
	clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
	clf.fit(X_train, Y_train)

	# predict
	predictions = clf.predict(X_dev)
	errors = sum([predictions[i] != Y_dev[i] for i in range(len(Y_dev))])
	confusion = confusion_matrix(predictions, Y_dev, labels = range(4))
	
	f = open('%s.txt' % accuracyFileName, 'a')
	print >> f, 'Dev error rate: ', errors/float(len(dev))
	f.close()

	print 'Dev error rate: ', errors/float(len(dev))

	f = open('%s.txt' % cmFileName, 'a') 
	print >> f, confusion
	f.close()

	print confusion 

