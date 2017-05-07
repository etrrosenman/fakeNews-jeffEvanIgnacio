import csv
from datetime import date, time, datetime
from pdb import set_trace as t 

# up the size of the filed
csv.field_size_limit(10000000)

short = 0 
with open('../Datasets/fake.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if len(row[4].split() + row[5].split()) < 20:
			short += 1

print short