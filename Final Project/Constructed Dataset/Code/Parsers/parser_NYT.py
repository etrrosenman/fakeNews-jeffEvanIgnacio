import re
import csv
import sys
from datetime import date, time, datetime
from string import rstrip
from pdb import set_trace as t 
from os import listdir


# prepare to store everything
results = list()

# get list of files and iterate through them
files = listdir(sys.argv[1])
for file in files:

	# go through the file
	with open(sys.argv[1] + file, 'rb') as f:

		# pre-process the file
		lines = f.readlines()
		lines = map(rstrip, lines)
		lines = [l for l in lines if len(l) > 0]

		# iterate through the nonzero length lines
		active = False 
		articleMap = [False for i in range(6)]

		# iterate through the lines
		for line in lines: 

			# condition to match the start of an article 
			if re.match("[0-9]+ of [0-9]+ DOCUMENTS", line.strip()):
				active = True
				results.append(dict())
				articleMap = [False for i in range(6)]
				continue

			# condition to match the end of an article
			if re.match("URL:", line) or re.match("Related Articles", line.strip()) or re.match("LOAD-DATE: .*", line):
				active = False 
			
			# if an article is open
			if active: 

				# get publication
				if sum(articleMap) == 0:
					results[len(results)-1]["Publication"] = line.strip()
					articleMap[0] = True
					continue

				# get date 
				if sum(articleMap) == 1:
					results[len(results)-1]["Date"] = line.strip()
					articleMap[1] = True
					continue

				# get title (and possibly byline)
				if sum(articleMap) == 2:

					# skip the edition line 
					if(len(line) - len(line.strip()) > 20):
						continue

					# check if it's the byline or the section
					bylineMatch = re.match("BYLINE: (.*)", line.strip())
					sectionMatch = re.match("SECTION: (.*)", line.strip())

					# if not, match the title
					if not (bylineMatch or sectionMatch):
						if "Title" not in results[len(results) - 1].keys():
							results[len(results) - 1]["Title"] = line
						else:
							results[len(results) - 1]["Title"] += " " + line
						continue 
					# if so, update the byline and keep going
					elif bylineMatch:
						results[len(results) - 1]["Byline"] = bylineMatch.group(1)
						articleMap[2] = True
						continue
					elif sectionMatch:
						results[len(results) - 1]["Section"] = sectionMatch.group(1)
						articleMap[3] = True
						continue

				# if matched a byline but not a section 
				if articleMap[2] and not articleMap[3]: 

					# check if it's the section 
					sectionMatch = re.match("SECTION: (.*)", line.strip())

					# if not, append to byline 
					if not sectionMatch:
						results[len(results) - 1]["Byline"] += " " + line
					# if so, append to section  
					else: 
						results[len(results) - 1]["Section"] = sectionMatch.group(1)
						articleMap[3] = True 
						continue

				# if matched a section 
				if articleMap[3] and not articleMap[4]: 

					# check if it's the length
					lengthMatch = re.match("LENGTH: (.*)", line.strip())

					# if not, append to section 
					if not lengthMatch:
						results[len(results) - 1]["Section"] += " " + line
					# if so, make note
					else: 
						articleMap[4] = True 

				# get text
				if articleMap[4]: 

					# check if length or dateline
					lengthMatch = re.match("LENGTH: (.*)", line.strip())
					datelineMatch = re.match("DATELINE: (.*)", line.strip())
					
					# if neither, append to text
					if not (datelineMatch or lengthMatch):
						if "Text" not in results[len(results) - 1].keys():
							results[len(results) - 1]["Text"] = line
						else:
							results[len(results) - 1]["Text"] = results[len(results) - 1]["Text"] + " " + line

# print the results to a csv
with open(sys.argv[2], 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(results[0].keys())
    for article in results:

    	# ignore letters to the editor, sports, etc. 
    	if len(article) == 6:
	    	writer.writerow(article.values())


