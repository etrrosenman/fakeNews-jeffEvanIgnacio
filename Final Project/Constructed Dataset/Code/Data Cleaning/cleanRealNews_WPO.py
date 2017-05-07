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
	publicationIndex =  [i for i in range(len(header)) if header[i] == 'Publication'][0]
	sectionIndex =  [i for i in range(len(header)) if header[i] == 'Section'][0]

	for row in reader:
		# exclude blogs 
		if row[publicationIndex] != 'Washington Post Blogs':
			if 'sports' not in row[sectionIndex].lower():
				results.append(row)

# output the results to a csv
with open("../../Datasets/Datasets -- Cleaned/" + sys.argv[1] + ".csv", 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for article in results:
    	writer.writerow(article)
