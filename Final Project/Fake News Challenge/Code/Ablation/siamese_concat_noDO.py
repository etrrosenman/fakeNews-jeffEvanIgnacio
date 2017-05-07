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
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize

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

	return helper, train_data, dev_data, train, dev


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

	if len(sys.argv) < 2 or not (sys.argv[1] == "RNN" or sys.argv[1] == "GRU" or sys.argv[1] == "LSTM"):
		print "Error: must specify cell type!"
		exit()

	print 'What should the accuracy file name be?'
	accuracyFileName = raw_input()
	print 'What should the confusion matrix file name be?'
	cmFileName = raw_input()
	print 'What should the output file name be?'
	outputFile = raw_input()

	######################################
	##           get the data           ##	
	######################################

	# load in the data
	debug = False  
	if len(sys.argv) > 2 and sys.argv[2] == "debug":
		debug = True
	helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(debug)
	pretrained_embeddings = load_embeddings(helper, vocabPath = "../Vectors/gloveVocab.txt", 
		vectorPath = "../Vectors/glove.6B.200d.txt", wordFirst = True, embed_size = 200)
	Config.embed_size = pretrained_embeddings.shape[1]

	# for later
	neverOpened_gold = True 
	neverOpened_test = True 


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

	# get the cell 
	if sys.argv[1] == "RNN":
		cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
		cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
	elif sys.argv[1] == "GRU":
		cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
		cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
	elif sys.argv[1] == "LSTM":
		cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
		cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size), dropout_placeholder, dropout_placeholder)
	else:
		print "must specify cell type!"
		exit()

	# process the headline
	with tf.variable_scope('cell'):
		headline_output, headline_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
			headline_embedded_tensor, dtype = tf.float32, sequence_length = headline_lengths_placeholder)  

	# process the article
	with tf.variable_scope('cell', reuse=True):
		article_output, article_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
			article_embedded_tensor, dtype = tf.float32, sequence_length = article_lengths_placeholder)  


	# concatenate the tensors? 
	if sys.argv[1] != "LSTM":
		headline_state = tf.concat(1, [headline_states[0], headline_states[1]])
		article_state = tf.concat(1, [article_states[0], article_states[1]])
	else:
		headline_state = tf.concat(1, [headline_states[0][1], headline_states[1][1]])
		article_state = tf.concat(1, [article_states[0][1], article_states[1][1]])
	combined_hidden = tf.concat(1, [headline_state, article_state]) 

	# define variables for multiplying
	W = tf.get_variable('W', (2*Config.headline_hidden_size + 2*Config.article_hidden_size, 
		Config.final_state_size), initializer = tf.contrib.layers.xavier_initializer())
	b = tf.get_variable('b', Config.final_state_size, initializer = tf.zeros_initializer)

	#apply dropout rate here?
	combined_hidden = combined_hidden, dropout_placeholder

	# compute
	last_hidden = tf.nn.relu(tf.matmul(combined_hidden, W) + b)
	W2 = tf.get_variable('W2', (Config.final_state_size, Config.num_classes), 
		initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', Config.num_classes, initializer = tf.zeros_initializer)
	output = tf.matmul(last_hidden, W2) + b2

	total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels_placeholder)
	loss = tf.reduce_mean(total_loss)

	# after getting loss we run the training optimizer and then pass this into sess.run() to call the model to train
	train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)

	######################################
	##               train              ##	
	######################################

	# graph already created

	#attribution: https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23
	# create a session
	session = tf.Session() 

	# create the initialize operator and run it
	init = tf.global_variables_initializer()
	session.run(init) 

	# loop over epoch count
	for epoch in range(Config.epochs): 
		epoch_loss = 0
		batches = generate_batches(train)

		# training loop 
		batchCounter = 0 
		for batch in batches:
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
			headline_lengths_placeholder: headline_lengths, article_lengths_placeholder: article_lengths, dropout_placeholder: Config.dropout}
			if labels_batch is not None:
				feed_dict[labels_placeholder] = labels_batch

			# run session
			_, local_loss = session.run([train_op, loss], feed_dict = feed_dict)
			epoch_loss += local_loss

		# evaluation loop
		devBatches = generate_batches(dev)
		errors = 0.0
		confusion = np.zeros((4, 4))
		for batch in devBatches:
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
			confusion += confusion_matrix(predictions, labels_batch, labels = range(4))
			errors += sum([labels_batch[i] != predictions[i] for i in range(len(labels_batch))])

			# on last iteration, print out the results
			if epoch == Config.epochs - 1:
				with open('%s_gold.csv' % outputFile, 'a') as csv_file:
					writer = csv.writer(csv_file)
					if neverOpened_gold:
						writer.writerow(['Headline', 'Body ID', 'Stance'])
						neverOpened_gold = False
					for i in range(len(labels_batch)):
						writer.writerow(['','',LBLS[labels_batch[i]]])
				with open('%s_test.csv' % outputFile, 'a') as csv_file:
					writer = csv.writer(csv_file)
					if neverOpened_test:
						writer.writerow(['Headline', 'Body ID', 'Stance'])
						neverOpened_test = False
					for i in range(len(labels_batch)):
						writer.writerow(['','',LBLS[predictions[i]]])

		
		f = open('%s.txt' % accuracyFileName, 'a')
		print >> f, epoch
		print >> f, 'Training loss: ', epoch_loss/float(batchCounter)
		print >> f, 'Dev error rate: ', errors/float(len(dev))
		f.close()

		f = open('%s.txt' % cmFileName, 'a') 
		print >> f, epoch
		print >> f, confusion
		f.close()

