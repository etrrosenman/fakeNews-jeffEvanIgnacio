import re
import csv
import sys
import util 
from datetime import date, time, datetime
from string import rstrip
from pdb import set_trace as t 
from os import listdir


# prepare to store everything
results = list()
header = None 

# temp
sectionCounts = {}

# go through the file
with open("../../Datasets/Datasets -- No Cleaning/" + sys.argv[1] + ".csv", 'rU') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	header = next(reader, None) # skip the header 
	sectionIndex =  [i for i in range(len(header)) if header[i] == 'Section'][0]

	for row in reader:
		if (('sports' in row[sectionIndex].lower()) or 
		('arts' in row[sectionIndex].lower()) or
		('style' in row[sectionIndex].lower()) or
		('oped' in row[sectionIndex].lower()) or
		('opinion' in row[sectionIndex].lower()) or
		('leisure' in row[sectionIndex].lower()) or
		('review' in row[sectionIndex].lower()) or
		('obituary' in row[sectionIndex].lower()) or
		('editorial' in row[sectionIndex].lower())):
			continue 
		else:
			# split by sentence and delete last sentence (frequently noise)
			sentences = row[0].lower().split(".")
			row[0] = ".".join(sentences[1:(len(sentences) - 2)])

			# delete obvious strings 
			row[0] = row[0].replace(' pp.', '')

			results.append(row)

# output the results to a csv
with open("../../Datasets/Datasets -- Cleaned/" + sys.argv[1] + ".csv", 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for article in results:
    	writer.writerow(article)
