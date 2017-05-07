import re
import csv
from datetime import date, time, datetime
from string import rstrip
from pdb import set_trace as t 

# go through the file
with open('USA_Today_Los_Angeles_Times_CNN_com_Daily_Ne2017-02-17_22-15.TXT', 'rb') as f:
	
	# pre-process the file
	lines = f.readlines()
	lines = map(rstrip, lines)
	lines = [l for l in lines if len(l) > 0]

	# iterate through the nonzero length lines
	active = False 
	counter = 0
	results = list()

	# iterate through the lines
	for line in lines: 

		# condition to match the start of an article 
		if re.match("[0-9]+ of [0-9]+ DOCUMENTS", line.strip()):
			active = True
			counter = 1 
			results.append(dict())
			continue

		# condition to match the end of an article
		if re.match("LOAD-DATE: .*", line):
			active = False 
		
		# if an article is open
		if active: 
			# get publication
			if counter == 1:
				results[len(results)-1]["Publication"] = line.strip()
			# get date
			elif counter == 2:
				results[len(results)-1]["Date"] = line.strip()
			# get title 
			elif counter == 3:
				# if the line starts with a lot of whitespace, it's a continuation
				# of the prior line (says the edition)
				if(len(line) - len(line.strip()) > 20):
					continue
				results[len(results) - 1]["Title"] = line.strip()
			# either get second half of title or byline 
			elif counter == 4:
				# match the byline 
				bylineMatch = re.match("BYLINE: (.*)", line.strip())
				if bylineMatch: 
					results[len(results) - 1]["Byline"] = bylineMatch.group(1)
				else:
					results[len(results) - 1]["Title"] += " " + line.strip()
			elif counter > 4: 

				# match the byline 
				bylineMatch = re.match("BYLINE: (.*)", line.strip())
				if bylineMatch: 
					results[len(results) - 1]["Byline"] = bylineMatch.group(1)

				# match the section
				sectionMatch = re.match("SECTION: (.*)", line.strip())
				if sectionMatch: 
					results[len(results) - 1]["Section"] = sectionMatch.group(1)

				# if not byline, section, length, or dateline, it's text
				datelineMatch = re.match("DATELINE: (.*)", line.strip())
				lengthMatch = re.match("LENGTH: (.*)", line.strip())
				if not (bylineMatch or sectionMatch or datelineMatch or lengthMatch):
					if "Text" not in results[len(results) - 1].keys():
						results[len(results) - 1]["Text"] = line
					else:
						results[len(results) - 1]["Text"] = results[len(results) - 1]["Text"] + " " + line

			counter += 1 			
	
	# print the results to a csv
	with open('articles.csv', 'wb') as csv_file:
	    writer = csv.writer(csv_file)
	    writer.writerow(results[0].keys())
	    for article in results:
	    	writer.writerow(article.values())


