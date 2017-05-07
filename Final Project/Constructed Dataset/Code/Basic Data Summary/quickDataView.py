import csv
from datetime import date, time, datetime
from pdb import set_trace as t 

# up the size of the filed
csv.field_size_limit(10000000)

# iterate through the file
count = 0 
minDate = datetime.now()
maxDate = datetime(2016, 1, 1)
sources = {}

with open('fake.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if ("Clinton" in row[5]) or ("Trump" in row[5]) or ("clinton" in row[5]) or ("trump" in row[5]) or ("election" in row[5]): 
			if row[8] in sources.keys():
				sources[row[8]] = sources[row[8]] + 1
			else:
				sources[row[8]] = 1
			count += 1
			date = datetime.strptime(row[3].split("+")[0], "%Y-%m-%dT%H:%M:%S.%f")
			if date < minDate: 
				minDate = date
			if date > maxDate:
				maxDate = date

print count 
print minDate
print maxDate
print len(sources) 

relCount = 0 
relTopicCount = 0 
with open('Katharina Data.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile, delimiter = ",", dialect=csv.excel_tab)
	next(reader, None) # skip the header 
	for row in reader:
		# see if date matches 
		try:
			kDate = datetime.strptime(row[1], "%m/%d/%y")
		except ValueError:
			continue
		
		# check if date in region 
		if kDate > minDate and kDate < maxDate: 
			relCount += 1
			if ("Clinton" in row[9]) or ("Trump" in row[9]) or ("clinton" in row[9]) or ("trump" in row[9]) or ("election" in row[9]): 
				relTopicCount += 1 

print relCount 
print relTopicCount


