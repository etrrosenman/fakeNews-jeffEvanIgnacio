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

#Jeff added this to pad sequences
def pad_sequences(data, max_length_headline, max_length_articles):
	ret = []
	lengths = []

	# Use this zero vector when padding sequences.
	zero_vector = 0 #* Config.n_features

	for headline, article, labels in data:
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
		one_data_point = (copy_headline, copy_article, labels)
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
	
	data = [(cnnHeadlines, cnnArticles), (nytHeadlines, nytArticles), (usaHeadlines, usaArticles), \
			(latHeadlines, latArticles), (wapoHeadlines, wapoArticles), (fakeHeadlines, fakeArticles)]

	for u, v in data:
		print len(u), np.mean([len(h.split()) for h in u]), np.mean([len(a.split()) for a in v]), \
			np.percentile([len(a.split()) for a in v], 50), np.percentile([len(a.split()) for a in v], 10),\
			np.percentile([len(a.split()) for a in v], 90)

	t()

	# look at fake headlines with their actual headlines
	fake = [(fakeHeadlines[i][:40], fakeArticles[i][:40], 'discuss') for i in fakeIndices]
	usa = [(usaHeadlines[i][:40], usaArticles[i][:40], 'discuss') for i in usaIndices]
	wapo = [(wapoHeadlines[i][:40], wapoArticles[i][:40], 'discuss') for i in wapoIndices]
	cnn = [(cnnHeadlines[i][:40], cnnArticles[i][:40], 'discuss') for i in cnnIndices]
	nyt = [(nytHeadlines[i][:40], nytArticles[i][:40], 'discuss') for i in nytIndices]
	lat = [(latHeadlines[i][:40], latArticles[i][:40], 'discuss') for i in latIndices]

	sample = fake + usa + wapo + cnn + nyt + lat

	# vectorize everything 
	fakeVec = helper.vectorize(fake)
	usaVec = helper.vectorize(usa)
	wapoVec = helper.vectorize(wapo)
	cnnVec = helper.vectorize(cnn)
	nytVec = helper.vectorize(nyt)
	latVec = helper.vectorize(lat)

	fakeVec = [(a[0], a[1], 0) for a in fakeVec]
	usaVec = [(a[0], a[1], 1) for a in usaVec]
	wapoVec = [(a[0], a[1], 2) for a in wapoVec]
	cnnVec = [(a[0], a[1], 3) for a in cnnVec]
	nytVec = [(a[0], a[1], 4) for a in nytVec]
	latVec = [(a[0], a[1], 5) for a in latVec]
	allData = fakeVec + usaVec + wapoVec + cnnVec + nytVec + latVec
	
	return helper, allData, sample 


def generate_batches(data):

	# shuffle the data
#	np.random.seed(2017)
#	np.random.shuffle(data)

	# create batches
	batchNum = int(np.ceil(len(data)/float(Config.batch_size)))
	batches = []
	for i in range(batchNum):
		base = i*Config.batch_size
		batches.append(data[base:(min(base + Config.batch_size, len(data)))])
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
	devBatches = generate_batches(eval)
	results = []
#	results = np.zeros((2, 4))
	batchCounter = 0 
	for batch in devBatches:
		batchCounter = batchCounter + 1
		if batchCounter % 20 == 0:
			print batchCounter 

		batch, lengths = pad_sequences(batch, Config.max_length_titles, Config.max_length_articles)
		headline_inputs_batch = np.array([x[0] for x in batch])
		article_inputs_batch = np.array([x[1] for x in batch])
		labels_batch = np.array([x[2] for x in batch])
		headline_lengths = np.array([x[0] for x in lengths])
		article_lengths = np.array([x[1] for x in lengths])

		# create_feed_dict
		feed_dict = {headline_input_placeholder: headline_inputs_batch, article_input_placeholder: article_inputs_batch,
		headline_lengths_placeholder: headline_lengths, article_lengths_placeholder: article_lengths, dropout_placeholder: 1.0}

		# run session
		predictions = session.run(tf.argmax(output, axis=1), feed_dict=feed_dict)
		results.extend(zip(predictions, labels_batch))
#		for i in range(len(predictions)):
#			results[labels_batch[i]][predictions[i]] +=1 
	with open('./comparison_to_own_headline.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for row in results:
			writer.writerow(row)
		




