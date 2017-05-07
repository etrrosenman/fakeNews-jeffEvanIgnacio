import re
import csv
import nltk
from datetime import date, time, datetime
from string import rstrip
from pdb import set_trace as t 

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


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

# make able to look at larger data
csv.field_size_limit(10000000)


# go through the file
with open('fake.csv', 'rb') as csvFile:

	reader = csv.reader(csvFile, delimiter=',')
	next(reader, None) # skip the header 

	# prepare to store the named entities and counts
	namedEntities = {}

	# go through the document 
	counter = 0 
	for row in reader:

		text = unicode(row[5], errors='replace')

		# keep track of progress
		if counter % 20 == 0:
			print counter

		# tokenize, tag, and pull NEs from data
		tokens = nltk.word_tokenize(text)
		tags = nltk.pos_tag(tokens)
		chunks = nltk.ne_chunk(tags, binary = True)

		# get a list of named entities (this is inefficient, but oh well)
		NE = []
		for c in chunks:
			NE.extend(extract_entity_names(c))

		# put everything in the dictionary
		for named in NE:
			if named not in namedEntities.keys():
				namedEntities[named] = 1
			else:
				namedEntities[named] += 1

		counter += 1

	# print the results to a file
	with open('namedEntities.csv', 'wb') as csv_file:

		# define fields
		fieldnames = ['entity', 'count']
		writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
		writer.writeheader()

		# print the entry
		for entry in namedEntities:
			writer.writerow({'entity': entry, 'count':namedEntities.get(entry)})



