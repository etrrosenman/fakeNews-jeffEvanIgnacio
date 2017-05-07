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

# go through the file
with open("../../Datasets/Datasets -- No Cleaning/fake.csv", 'rb') as csvfile:

	reader = csv.reader(csvfile, delimiter=',')
	header = next(reader, None) # skip the header 

	for row in reader:
		date = re.match("(.*)T", row[3])
		newDate = datetime.strptime(date.group(1), "%Y-%m-%d").strftime('%m/%-d/%y')
		newRow = row[:3] + [newDate] + row[4:]

		# will this help get rid of bad characters? 
		newRow[5] = newRow[5].decode('utf-8')
		newRow[5] = newRow[5].encode('ascii', 'ignore')
		newRow[5] = newRow[5].decode('utf-8')

		if row[6] == 'english':
			results.append(newRow)

		# article = util.clean(article)
		# if article.endswith('More Related'):
		# 	article = article[:-13]
		# if article.endswith('Related'):
		# 	article = article[:-7]
		# if article.endswith('TRENDING ON Fed Up'):
		# 	article = article[:-18]
		# if article.startswith('Print') or article.startswith('Email'):
		# 	article = article[6:]
		# if article.startswith('Red State '):
		# 	article = article[10:]

# output the results to a csv
with open("../../Datasets/Datasets -- Cleaned/fake.csv", 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for article in results:
    	writer.writerow(article)
