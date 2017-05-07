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


def preprocess_sequence_data(self, examples):
	def featurize_windows(data, start, end, window_size = 1):
		"""Uses the input sequences in @data to construct new windowed data points."""
		ret = []
		for sentence, labels in data:
			from util import window_iterator
			sentence_ = []
			for window in window_iterator(sentence, window_size, beg=start, end=end):
				sentence_.append(sum(window, []))
			ret.append((sentence_, labels))
		return ret

	examples = featurize_windows(examples, self.helper.START, self.helper.END)
	return pad_sequences(examples, self.max_length_titles)

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

#class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
#	def __init__(self, num_units, encoder_output, scope = None):
#		self.hs = encoder_output
#		super(GRUAttnCell, self).__init__(num_units)
#
#	def __call__(self, inputs, state, scope = None):
#		gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
#		with tf.variable_scope(scope or type(self).__name__):
#			with tf.variable_scope("Attn"):
#				ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)
#				ht = tf.expand_dims(ht, axis = 1)
#			scores = tf.reduce_sum(self.hs * ht, reduction_indices = 2, keep_dims = True)
#			scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=0, keep_dims=True))
#			scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=0, keep_dims=True))
#			context = tf.reduce_sum(self.hs * scores, reduction_indices = 1)
#			with tf.variable_scope("AttnConcat"):
#				out = tf.nn.relu(tf.nn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
#		return (out, out)

class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
	def __init__(self, num_units, encoder_output, scope=None):
		self.hs = encoder_output
		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn1"):
				hs2d = tf.reshape(self.hs, [-1, num_units])
				phi_hs2d = tf.nn.tanh(tf.nn.rnn_cell._linear(hs2d, num_units, True, 1.0))
				self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
			super(GRUAttnCell, self).__init__(num_units)
	def __call__(self, inputs, state, scope=None):
		gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn2"):
				gamma_h = tanh(tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0))
			weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
			weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
			weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
			context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
			with tf.variable_scope("AttnConcat"):
				out = tf.nn.relu(tf.nn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
			self.attn_map = tf.squeeze(tf.slice(weights, [0, 0, 0], [-1, -1, 1]))
		return (out, out)	

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
	##           get the graph          ##	
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

	# get the cell for articles ONLY in GRU, note, this code only works for GRU - do not pass others
	
	#if sys.argv[1] == "RNN":
	#	headline_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size)
	#	headline_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.headline_hidden_size)
	#elif sys.argv[1] == "GRU":
	article_cell_fw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
	article_cell_bw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
	#elif sys.argv[1] == "LSTM":
	#	headline_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size)
	#	headline_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.headline_hidden_size)
	#else:
	#	print "must specify cell type!"
	#	exit()

	# create the article cell
	with tf.variable_scope('article_cell') as vs:
		article_output, article_states = tf.nn.bidirectional_dynamic_rnn(article_cell_fw, article_cell_bw, 
			article_embedded_tensor, dtype = tf.float32, sequence_length = article_lengths_placeholder)  
		# variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
		# for v in variables:
		# 	tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)

	# get the cell for headlines - NOTE: only GRU will work here
	
	#if sys.argv[1] == "RNN":
	#	article_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
	#	article_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
	#elif sys.argv[1] == "GRU":
	headline_cell_fw = GRUAttnCell(Config.headline_hidden_size, article_output[0])
	headline_cell_bw = GRUAttnCell(Config.headline_hidden_size, article_output[1])
	#elif sys.argv[1] == "LSTM":
	#	article_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)
	#	article_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)

	# create the headline cell
	with tf.variable_scope('headline_cell'):
		headline_output, headline_states = tf.nn.bidirectional_dynamic_rnn(headline_cell_fw, headline_cell_bw, 
			headline_embedded_tensor, dtype = tf.float32, sequence_length = headline_lengths_placeholder,
			initial_state_fw = article_states[0], initial_state_bw = article_states[1])  

	# first pass: add the tensors? 
	#if sys.argv[1] != "LSTM":
	article_state = article_states[0] + article_states[1]
	#else:
	#	article_state = article_states[0][1] + article_states[1][1]
	combined_hidden = article_state

	# define variables for multiplying
	W = tf.get_variable('W', (Config.article_hidden_size, Config.final_state_size), 
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

#	output = tf.clip_by_value(output,1e-10,100.0)

	total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels_placeholder)
	loss = tf.reduce_mean(total_loss)

	# after getting loss we run the training optimizer and then pass this into sess.run() to call the model to train	train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
	train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
	# tvars = tf.trainable_variables()
	# grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), .00001)
	# train_op = optimizer.apply_gradients(zip(grads, tvars))

#	gvs = optimizer.compute_gradients(loss)
#	capped_gvs = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in gvs]
#	train_op = optimizer.apply_gradients(capped_gvs)


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


	# create a session 
	# inside the session: create an init operator that initializes all the variables
	# loop over the number of epochs that we want to do
		# within each epoch, loop over the batches (need a function of randomly shuffled batches --
		# extract batches from the array. look at previous assignments code) 

		# call sess.run() and call the trainOp 




