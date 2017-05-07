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
	conditional_hidden_size = 50 #originally 100
	numPerspectives = 10

	batch_size = 32
	embed_size = 300 # try a larger embedding
	max_length_titles = 40
	max_length_articles = 100 #changed from 700
	num_classes = 4
	epochs = 10
	lr = 0.001
	dropout = 1.0
	final_state_size = 20 


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


	#temp
	allHeadlines = [v['Headline'] for v in stances]
	v = {}
	for a in allHeadlines:
		if a in v.keys():
			v[a] += 1
		else:
			v[a] = 1
	t()

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
	batch_size_placeholder = tf.placeholder(tf.int32, shape = (None,))

	# dropout placeholder:
	dropout_placeholder = tf.placeholder(tf.float32, shape = ())

	# create an embeddings variable
	embedding = tf.Variable(pretrained_embeddings, trainable = False)
	headline_embedded_tensor = tf.nn.embedding_lookup(embedding, headline_input_placeholder)
	article_embedded_tensor = tf.nn.embedding_lookup(embedding, article_input_placeholder)

	# get the cell for encoding both articles and headlines 
	if sys.argv[1] == "RNN":
		context_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
		context_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.article_hidden_size)
	elif sys.argv[1] == "GRU":
		context_cell_fw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
		context_cell_bw = tf.nn.rnn_cell.GRUCell(Config.article_hidden_size)
	elif sys.argv[1] == "LSTM":
		context_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)
		context_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.article_hidden_size)

	# process the article
	with tf.variable_scope('context_rep'):
		article_output, article_states = tf.nn.bidirectional_dynamic_rnn(context_cell_fw, context_cell_bw, 
			article_embedded_tensor, dtype = tf.float32, sequence_length = article_lengths_placeholder, 
			time_major = False, swap_memory = True)  

	# process the headline 
	with tf.variable_scope('context_rep', reuse = True):
		headline_output, headline_states = tf.nn.bidirectional_dynamic_rnn(context_cell_fw, context_cell_bw, 
			headline_embedded_tensor, dtype = tf.float32, sequence_length = headline_lengths_placeholder,
			time_major = False, swap_memory = True)  

	# multi perspective matching layer
	normalized_headline_forward = tf.nn.l2_normalize(headline_output[0], dim = 2)
	normalized_headline_backward = tf.nn.l2_normalize(headline_output[1], dim = 2)
	normalized_article_forward = tf.nn.l2_normalize(article_output[0], dim = 2)
	normalized_article_backward = tf.nn.l2_normalize(article_output[1], dim = 2)
	forward_match = tf.batch_matmul(normalized_headline_forward, 
		tf.transpose(normalized_article_forward, [0, 2, 1]))
	backward_match = tf.batch_matmul(normalized_headline_forward, 
		tf.transpose(normalized_article_forward, [0, 2, 1]))
	
	# compute all the normalized context vectors 
	headline_to_article_forward_context = tf.batch_matmul(forward_match, article_output[0])
	headline_to_article_forward_context = headline_to_article_forward_context/\
		tf.tile(tf.expand_dims(tf.reduce_sum(forward_match, 2), 2), [1, 1, Config.article_hidden_size])
	headline_to_article_backward_context = tf.batch_matmul(backward_match, article_output[1])
	headline_to_article_backward_context = headline_to_article_backward_context/\
		tf.tile(tf.expand_dims(tf.reduce_sum(backward_match, 2), 2), [1, 1, Config.article_hidden_size])

	article_to_headline_forward_context = tf.matmul(tf.transpose(forward_match, [0, 2, 1]), headline_output[0])
	article_to_headline_forward_context = article_to_headline_forward_context/\
		tf.tile(tf.expand_dims(tf.reduce_sum(forward_match, 1), 2), [1, 1, Config.headline_hidden_size])
	article_to_headline_backward_context = tf.matmul(tf.transpose(backward_match, [0, 2, 1]), headline_output[0])
	article_to_headline_backward_context = article_to_headline_backward_context/\
		tf.tile(tf.expand_dims(tf.reduce_sum(backward_match, 1), 2), [1, 1, Config.headline_hidden_size])

	# do the multi perspective matching 
	h_to_a_W1 = tf.get_variable('h_to_a_W1', 
		(Config.numPerspectives, Config.headline_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	h_to_a_W2 = tf.get_variable('h_to_a_W2', 
		(Config.numPerspectives, Config.headline_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	a_to_h_W1 = tf.get_variable('a_to_h_W1', 
		(Config.numPerspectives, Config.article_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	a_to_h_W2 = tf.get_variable('a_to_h_W2', 
		(Config.numPerspectives, Config.article_hidden_size), initializer = tf.contrib.layers.xavier_initializer())

	h_to_a_W5 = tf.get_variable('h_to_a_W5', 
		(Config.numPerspectives, Config.headline_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	h_to_a_W6 = tf.get_variable('h_to_a_W6', 
		(Config.numPerspectives, Config.headline_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	a_to_h_W5 = tf.get_variable('a_to_h_W5', 
		(Config.numPerspectives, Config.article_hidden_size), initializer = tf.contrib.layers.xavier_initializer())
	a_to_h_W6 = tf.get_variable('a_to_h_W6', 
		(Config.numPerspectives, Config.article_hidden_size), initializer = tf.contrib.layers.xavier_initializer())

	h_to_a_Perspectives = []
	a_to_h_Perspectives = []
	for i in range(Config.numPerspectives):
		h_to_a_f1_1 = tf.nn.l2_normalize(tf.multiply(h_to_a_W1[i], headline_output[0]), dim = 2)
		h_to_a_f2_1 = tf.nn.l2_normalize(tf.multiply(h_to_a_W1[i], article_states[0][1]), dim = 1)
		h_to_a_Perspectives.append(tf.expand_dims(tf.reduce_sum(\
			tf.transpose(tf.multiply(tf.transpose(h_to_a_f1_1, perm = [1, 0, 2]), h_to_a_f2_1), [1, 0, 2]), axis = 2), 2))

		h_to_a_f1_3 = tf.nn.l2_normalize(tf.multiply(h_to_a_W5[i], headline_output[0]), dim = 2)
		h_to_a_f2_3 = tf.nn.l2_normalize(tf.multiply(h_to_a_W5[i], headline_to_article_forward_context), dim = 2)
		h_to_a_Perspectives.append(tf.expand_dims(tf.reduce_sum(tf.multiply(h_to_a_f1_3, h_to_a_f2_3), axis = 2), 2))

		h_to_a_b1_1 = tf.nn.l2_normalize(tf.multiply(h_to_a_W2[i], headline_output[1]), dim = 2)
		h_to_a_b2_1 = tf.nn.l2_normalize(tf.multiply(h_to_a_W2[i], article_states[1][1]), dim = 1)
		h_to_a_Perspectives.append(tf.expand_dims(tf.reduce_sum(\
			tf.transpose(tf.multiply(tf.transpose(h_to_a_b1_1, perm = [1, 0, 2]), h_to_a_b2_1), [1, 0, 2]), axis = 2), 2))

		h_to_a_b1_3 = tf.nn.l2_normalize(tf.multiply(h_to_a_W6[i], headline_output[1]), dim = 2)
		h_to_a_b2_3 = tf.nn.l2_normalize(tf.multiply(h_to_a_W6[i], headline_to_article_backward_context), dim = 2)
		h_to_a_Perspectives.append(tf.expand_dims(tf.reduce_sum(tf.multiply(h_to_a_b1_3, h_to_a_b2_3), axis = 2), 2))

		a_to_h_f1_1 = tf.nn.l2_normalize(tf.multiply(a_to_h_W1[i], article_output[0]), dim = 2)
		a_to_h_f2_1 = tf.nn.l2_normalize(tf.multiply(a_to_h_W1[i], headline_states[0][1]), dim = 1)
		a_to_h_Perspectives.append(tf.expand_dims(tf.reduce_sum(\
			tf.transpose(tf.multiply(tf.transpose(a_to_h_f1_1, perm = [1, 0, 2]), a_to_h_f2_1), [1, 0, 2]), axis = 2), 2))		

		a_to_h_f1_3 = tf.nn.l2_normalize(tf.multiply(a_to_h_W5[i], article_output[0]), dim = 2)
		a_to_h_f2_3 = tf.nn.l2_normalize(tf.multiply(a_to_h_W5[i], article_to_headline_forward_context), dim = 2)
		a_to_h_Perspectives.append(tf.expand_dims(tf.reduce_sum(tf.multiply(a_to_h_f1_3, a_to_h_f2_3), axis = 2), 2))

		a_to_h_b1_1 = tf.nn.l2_normalize(tf.multiply(a_to_h_W2[i], article_output[1]), dim = 2)
		a_to_h_b2_1 = tf.nn.l2_normalize(tf.multiply(a_to_h_W2[i], headline_states[1][1]), dim = 1)
		a_to_h_Perspectives.append(tf.expand_dims(tf.reduce_sum(\
			tf.transpose(tf.multiply(tf.transpose(a_to_h_b1_1, perm = [1, 0, 2]), a_to_h_b2_1), [1, 0, 2]), axis = 2), 2))		

		a_to_h_b1_3 = tf.nn.l2_normalize(tf.multiply(a_to_h_W6[i], article_output[1]), dim = 2)
		a_to_h_b2_3 = tf.nn.l2_normalize(tf.multiply(a_to_h_W6[i], article_to_headline_backward_context), dim = 2)
		a_to_h_Perspectives.append(tf.expand_dims(tf.reduce_sum(tf.multiply(a_to_h_b1_3, a_to_h_b2_3), axis = 2), 2))

	h_to_aVec = tf.concat(2, h_to_a_Perspectives)
	a_to_hVec = tf.concat(2, a_to_h_Perspectives)

	# get the cells for the aggregation layer 
	if sys.argv[1] == "RNN":
		aggregation_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.conditional_hidden_size)
		aggregation_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.conditional_hidden_size)
	elif sys.argv[1] == "GRU":
		aggregation_cell_fw = tf.nn.rnn_cell.GRUCell(Config.conditional_hidden_size)
		aggregation_cell_bw = tf.nn.rnn_cell.GRUCell(Config.conditional_hidden_size)
	elif sys.argv[1] == "LSTM":
		aggregation_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.conditional_hidden_size)
		aggregation_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.conditional_hidden_size)

	# processs the conditional cell 
	with tf.variable_scope('aggregation'):
		agg_h_to_a_output, agg_h_to_a_states = tf.nn.bidirectional_dynamic_rnn(aggregation_cell_fw, aggregation_cell_bw, 
			h_to_aVec, dtype = tf.float32, sequence_length = headline_lengths_placeholder, time_major = False, swap_memory = True)
	with tf.variable_scope('aggregation', reuse = True):
		agg_a_to_h_output, agg_a_to_h_states = tf.nn.bidirectional_dynamic_rnn(aggregation_cell_fw, aggregation_cell_bw, 
			a_to_hVec, dtype = tf.float32, sequence_length = article_lengths_placeholder, time_major = False, swap_memory = True)

	# define variables for multiplying
	conditional_output_concat = tf.concat(1, [agg_a_to_h_states[0][1], agg_a_to_h_states[1][1], 
		agg_h_to_a_states[0][1], agg_h_to_a_states[1][1]])
	W = tf.get_variable('W', (4*Config.conditional_hidden_size, Config.final_state_size), 
		initializer = tf.contrib.layers.xavier_initializer())
	b = tf.get_variable('b', Config.final_state_size, initializer = tf.zeros_initializer)

	# compute
	last_hidden = tf.nn.relu(tf.matmul(conditional_output_concat, W) + b)
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
			batch_length = len(batch) #length of batch per
			batch_sizes = [Config.max_length_titles] * batch_length


			# create_feed_dict
			feed_dict = {headline_input_placeholder: headline_inputs_batch, article_input_placeholder: article_inputs_batch,
			headline_lengths_placeholder: headline_lengths, article_lengths_placeholder: article_lengths, dropout_placeholder: Config.dropout, 
			batch_size_placeholder: batch_sizes}
			if labels_batch is not None:
				feed_dict[labels_placeholder] = labels_batch

			# run session
			_, local_loss = session.run([train_op, loss], feed_dict = feed_dict)
			epoch_loss += local_loss
#			print epoch_loss/float(batchCounter), local_loss, batchCounter 

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
			batch_length = len(batch) #length of batch per
			batch_sizes = [Config.max_length_titles] * batch_length

			# create_feed_dict
			feed_dict = {headline_input_placeholder: headline_inputs_batch, article_input_placeholder: article_inputs_batch,
			headline_lengths_placeholder: headline_lengths, article_lengths_placeholder: article_lengths, dropout_placeholder: 1.0,
			batch_size_placeholder: batch_sizes}

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




