from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime
import re
import csv
import nltk
from datetime import date, time, datetime
from string import rstrip
from pdb import set_trace as t 

import tensorflow as tf
import numpy as np
import csv 
import articleUtils

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from defs import LBLS
from utils.dataset import DataSet
from sklearn.metrics import confusion_matrix
# from nltk.tokenize import word_tokenize
# from ner_model import NERModel

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

# thanks to https://gist.github.com/onyxfish/322906
def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

#Jeff added this to pad sequences
def pad_sequences(data, max_length_headline, max_length_articles):
	ret = []
	lengths = []

	# Use this zero vector when padding sequences.
	zero_vector = 0 #* Config.n_features

	for headline, article, labels, articleText in data:
		# copy everything over
		copy_headline = headline[:]
		copy_article = article[:]

		# fix the headline
		headline_len = len(headline) 
		if headline_len >= max_length_headline:
			copy_headline = headline[0:max_length_headline]
		else:
			diff = max_length_headline - headline_len
			for i in range(diff): 
				copy_headline.append(zero_vector)

		# fix the article
		article_len = len(article) 
		if article_len >= max_length_articles:
			copy_article = article[0:max_length_articles]
		else:
			diff = max_length_articles - article_len
			for i in range(diff): 
				copy_article.append(zero_vector)
		one_data_point = (copy_headline, copy_article, labels, articleText)
		ret.append(one_data_point)
		lengths.append((min(headline_len, max_length_headline), min(article_len, max_length_articles)))

	return ret, lengths

def load_and_preprocess_data(debug = False):

	######################################
	##         get the articles         ##	
	######################################

	# load up the training data
	debug = False # remove this 
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

	# create the helper object
	helper = ModelHelper.build(allData)

	# load up the data
	cnnHeadlines, cnnArticles, cnnDates = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/CNN.csv", 2, 3, 5)
	nytHeadlines, nytArticles, nytDates = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/NYT.csv", 2, 3, 5)
	usaHeadlines, usaArticles, usaDates = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/USA.csv", 3, 0, 4)
	latHeadlines, latArticles, latDates  = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/LAT.csv", 2, 3, 5)
	wapoHeadlines, wapoArticles, wapoDates = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/WPO.csv", 2, 3, 5)
	fakeHeadlines, fakeArticles, fakeDates = articleUtils.loadArticles("../../../Constructed Dataset/Datasets/Datasets -- Cleaned/fake.csv", 4, 5, 3)

	fakeIndices = range(len(fakeHeadlines))
	wapoIndices = range(len(wapoHeadlines))
	usaIndices = range(len(usaHeadlines))
	cnnIndices = range(len(cnnHeadlines))
	nytIndices = range(len(nytHeadlines))
	latIndices = range(len(latHeadlines))

	# pull headlines and articles for cnn, wapo, and fake news in period before election 
	preElectionCNNArticles = [cnnArticles[i].split() for i in cnnIndices if datetime.strptime(cnnDates[i], '%m/%d/%y') < 
		datetime.strptime('11/8/16', '%m/%d/%y') and datetime.strptime(cnnDates[i], '%m/%d/%y')]

	preElectionCNNHeadlines = [cnnHeadlines[i].split() for i in cnnIndices if datetime.strptime(cnnDates[i], '%m/%d/%y') < 
		datetime.strptime('11/8/16', '%m/%d/%y') and datetime.strptime(cnnDates[i], '%m/%d/%y')]

	preElectionWaPoHeadlines = [wapoHeadlines[i].split() for i in wapoIndices if 
		datetime.strptime(wapoDates[i], '%m/%d/%y') < datetime.strptime('11/8/16', '%m/%d/%y')]

	preElectionWaPoArticles = [wapoArticles[i].split() for i in wapoIndices if 
		datetime.strptime(wapoDates[i], '%m/%d/%y') < datetime.strptime('11/8/16', '%m/%d/%y')]

	preElectionFakeHeadlines = [fakeHeadlines[i].split() for i in fakeIndices if 
		datetime.strptime(fakeDates[i], '%m/%d/%y') < datetime.strptime('11/8/16', '%m/%d/%y')]

	preElectionFakeArticles = [fakeArticles[i].split() for i in fakeIndices if 
		datetime.strptime(fakeDates[i], '%m/%d/%y') < datetime.strptime('11/8/16', '%m/%d/%y')]

	# make a dict of the IDs and their named entities
	namedEntities = {}
	for ID in range(len(preElectionWaPoHeadlines)):

		# tokenize, tag, and pull NEs from data
		tags = nltk.pos_tag(preElectionWaPoHeadlines[ID])
		chunks = nltk.ne_chunk(tags, binary = True)

		# get a list of named entities (this is inefficient, but oh well)
		NE = []
		for c in chunks:
			NE.extend(extract_entity_names(c))

		# put everything in the dictionary
		for named in NE:
			if named not in namedEntities.keys():
				namedEntities[named] = [ID]
			else:
				namedEntities[named] += [ID]

	# create the sample of fake news-real news relevant pairings
	sample1 = []
	np.random.shuffle(preElectionFakeArticles)
	preElectionFakeSample = preElectionFakeArticles[:2000]
	totalCounter = 0
	relevantCounter = 0 

	for fakeArticle in preElectionFakeSample:
		totalCounter += 1
		if totalCounter % 20 == 0:
			print totalCounter, relevantCounter, len(sample1)

		# tokenize, tag, and pull NEs from data
		tags = nltk.pos_tag(fakeArticle)
		chunks = nltk.ne_chunk(tags, binary = True)

		# get a list of named entities (this is inefficient, but oh well)
		NE = []
		for c in chunks: 
			NE.extend(extract_entity_names(c))

		# try to get a list of relevant articles
		relevantIDs = [namedEntities[ne] for ne in NE if ne in namedEntities.keys()]
		relevantIDs = [val for sublist in relevantIDs for val in sublist]
		relevantIDs = np.unique(relevantIDs).tolist()

		# if relevant articles are found
		if len(relevantIDs) > 0:
			# cap if too many relevant IDs
			if len(relevantIDs) > 100:
				np.random.shuffle(relevantIDs)
				relevantIDs = relevantIDs[:100]
			pairs = [(preElectionWaPoHeadlines[i], fakeArticle[:min(len(fakeArticle), 40)], 'unrelated') \
				for i in relevantIDs]
			sample1.extend(pairs)
			relevantCounter += 1

	# create the sample of real news-real news relevant pairings
	sample2 = []
	np.random.shuffle(preElectionCNNArticles)
	preElectionCNNArticles = preElectionCNNArticles[:min(2000, len(preElectionCNNArticles))]
	totalCounter = 0
	relevantCounter = 0 

	for cnnArticle in preElectionCNNArticles:
		totalCounter += 1
		if totalCounter % 20 == 0:
			print totalCounter, relevantCounter, len(sample2)

		# tokenize, tag, and pull NEs from data
		tags = nltk.pos_tag(cnnArticle)
		chunks = nltk.ne_chunk(tags, binary = True)

		# get a list of named entities (this is inefficient, but oh well)
		NE = []
		for c in chunks: 
			NE.extend(extract_entity_names(c))

		# try to get a list of relevant articles
		relevantIDs = [namedEntities[ne] for ne in NE if ne in namedEntities.keys()]
		relevantIDs = [val for sublist in relevantIDs for val in sublist]
		relevantIDs = np.unique(relevantIDs).tolist()

		# if relevant articles are found
		if len(relevantIDs) > 0:
			# cap if too many relevant IDs
			if len(relevantIDs) > 100:
				np.random.shuffle(relevantIDs)
				relevantIDs = relevantIDs[:100]
			pairs = [(preElectionWaPoHeadlines[i], cnnArticle[:min(len(cnnArticle), 40)], 'unrelated') \
				for i in relevantIDs]
			sample2.extend(pairs)
			relevantCounter += 1

	# prepare all the data 
	sample = sample1 + sample2
	allData1 = helper.vectorize(sample1)
	allData2 = helper.vectorize(sample2)
	allData1 = [(a[0], a[1], 0) for a in allData1]
	allData2 = [(a[0], a[1], 1) for a in allData2]
	allData = allData1 + allData2

	return helper, allData, sample 


def generate_batches(data, rawdata):

	# add the relevant article text 
	newData = [] 
	for i in range(len(data)): 
		temp = list(data[i])
		temp.append(rawdata[i][1])
		newData.append(tuple(temp))

	# create batches
	batchNum = int(np.ceil(len(newData)/float(Config.batch_size)))
	batches = []
	for i in range(batchNum):
		base = i*Config.batch_size
		batches.append(newData[base:(min(base + Config.batch_size, len(newData)))])
	return batches


if __name__ == "__main__":


	######################################
	##           get the data           ##	
	######################################

	# load in the data
	debug = False  
	if len(sys.argv) > 2 and sys.argv[2] == "debug":
		debug = True
	helper, eval, eval_raw = load_and_preprocess_data(debug)

	pretrained_embeddings = load_embeddings(helper, vocabPath = "../Vectors/gloveVocab.txt", 
		vectorPath = "../Vectors/glove.6B.200d.txt", wordFirst = True, embed_size = 200)
	Config.embed_size = pretrained_embeddings.shape[1]

	######################################
	##           define graph           ##	
	######################################

	# define placeholders 
	headline_input_placeholder = tf.placeholder(tf.int32, (None, Config.max_length_titles, ))
	article_input_placeholder = tf.placeholder(tf.int32, (None, Config.max_length_articles, ))
	labels_placeholder = tf.placeholder(tf.int32, (None, )) # do we need to specify num_classes
	headline_lengths_placeholder = tf.placeholder(tf.int32, (None,))
	article_lengths_placeholder = tf.placeholder(tf.int32, (None,))

	# dropout placeholder:
	dropout_placeholder = tf.placeholder(tf.float32, shape = ())

	# create an embeddings variable
	embedding = tf.Variable(pretrained_embeddings, trainable = False)
	headline_embedded_tensor = tf.nn.embedding_lookup(embedding, headline_input_placeholder)
	article_embedded_tensor = tf.nn.embedding_lookup(embedding, article_input_placeholder)

	# get the cell for headlines 
	if sys.argv[1] == "RNN":
		headline_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size)
		headline_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size)
	elif sys.argv[1] == "GRU":
		headline_cell_fw = tf.nn.rnn_cell.GRUCell(Config.headline_hidden_size)
		headline_cell_bw = tf.nn.rnn_cell.GRUCell(Config.headline_hidden_size)
	elif sys.argv[1] == "LSTM":
		headline_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size)
		headline_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size)
	else:
		print "must specify cell type!"
		exit()

	# create the headline cell
	with tf.variable_scope('cell'):
		headline_output, headline_states = tf.nn.bidirectional_dynamic_rnn(headline_cell_fw, headline_cell_bw, 
			headline_embedded_tensor, dtype = tf.float32, sequence_length = headline_lengths_placeholder)  

	# get the cell for articles 
	if sys.argv[1] == "RNN":
		article_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
		article_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
	elif sys.argv[1] == "GRU":
		article_cell_fw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
		article_cell_bw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
	elif sys.argv[1] == "LSTM":
		article_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)
		article_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)

	# create the article cell
	with tf.variable_scope('cell', reuse=True):
		article_output, article_states = tf.nn.bidirectional_dynamic_rnn(article_cell_fw, article_cell_bw, 
			article_embedded_tensor, dtype = tf.float32, sequence_length = article_lengths_placeholder)  


	# first pass: add the tensors? 
	if sys.argv[1] != "LSTM":
		headline_state = headline_states[0] + headline_states[1]
		article_state = article_states[0] + article_states[1]
	else:
		headline_state = headline_states[0][1] + headline_states[1][1]
		article_state = article_states[0][1] + article_states[1][1]
	combined_hidden = tf.concat(1, [headline_state, article_state]) 

	# define variables for multiplying
	W = tf.get_variable('W', (Config.headline_hidden_size + Config.article_hidden_size, Config.final_state_size), 
		initializer = tf.contrib.layers.xavier_initializer())
	b = tf.get_variable('b', Config.final_state_size, initializer = tf.zeros_initializer)

	#apply dropout rate here?
	combined_hidden = tf.nn.dropout(combined_hidden, dropout_placeholder)

	# compute
	last_hidden = tf.nn.relu(tf.matmul(combined_hidden, W) + b)
	W2 = tf.get_variable('W2', (Config.final_state_size, Config.num_classes), 
		initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', Config.num_classes, initializer = tf.zeros_initializer)
	output = tf.matmul(last_hidden, W2) + b2

	total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels_placeholder)
	loss = tf.reduce_mean(total_loss)

	# saver
	saver = tf.train.Saver() 

	# after getting loss we run the training optimizer and then pass this into sess.run() to call the model to train
	train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)

	######################################
	##               train              ##	
	######################################

	# create a session
	saver = tf.train.Saver()
	session = tf.Session() 
	saver.restore(session, '../Siamese Layer/Saved_Model/')

	# evaluation loop
	devBatches = generate_batches(eval, eval_raw)
	results = {}
	batchCounter = 0 
	for batch in devBatches:
		batchCounter = batchCounter + 1
		if batchCounter % 20 == 0:
			print batchCounter 

		batch, lengths = pad_sequences(batch, Config.max_length_titles, Config.max_length_articles)
		headline_inputs_batch = np.array([x[0] for x in batch])
		article_inputs_batch = np.array([x[1] for x in batch])
		labels_batch = np.array([x[2] for x in batch])
		article_text = np.array([x[3] for x in batch])
		headline_lengths = np.array([x[0] for x in lengths])
		article_lengths = np.array([x[1] for x in lengths])

		# create_feed_dict
		feed_dict = {headline_input_placeholder: headline_inputs_batch, article_input_placeholder: article_inputs_batch,
		headline_lengths_placeholder: headline_lengths, article_lengths_placeholder: article_lengths, dropout_placeholder: 1.0}

		# run session
		predictions = session.run(tf.argmax(output, axis=1), feed_dict=feed_dict)
		for i in range(len(predictions)):
			article = ' '.join(article_text[i])
			if (article, labels_batch[i]) in results.keys():
				temp = results[(article, labels_batch[i])][:]
				temp[predictions[i]] += 1
				results[(article, labels_batch[i])] = temp
			else:
				temp = [0, 0, 0, 0]
				temp[predictions[i]] += 1
				results[(article, labels_batch[i])] = temp

	resultsList = [(results[v], v[1], v[0]) for v in results.keys()]
	with open('./results_namedEntityMatching.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for row in resultsList:
			a = row[0]
			a.append(row[1])
			writer.writerow(a)
		




