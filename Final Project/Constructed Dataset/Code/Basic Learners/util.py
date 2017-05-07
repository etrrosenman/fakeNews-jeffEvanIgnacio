import csv
import string
from pdb import set_trace as t
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

# up the size of the filed
csv.field_size_limit(10000000)

def loadArticles(csvPath, isFake, articleCol, dateCol, delNums = True, rnw = True):
    """
    Load all articles from a given file 
    """    

    articles = [] 
    dates = []
    labels = []

    with open(csvPath, 'rU') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None) # skip the header 

        counter = 0

        for row in reader:

            if counter % 50 == 0:
                print counter

            # kill all non-unicode articles (foreign language -- may want better 
            # way to deal with this in the future)
            try:
                article = row[articleCol]
                article = article.decode('utf-8', 'ignore')

                # tokenize the article
                article = sent_tokenize(article)

                # delete numbers 
                if delNums:                
                    chars_to_remove = '1234567890'
                    table = {ord(char): None for char in chars_to_remove}
                    article = [a.translate(table).encode('ascii', 'ignore') for a in article]

                if rnw:                
#                    chars_to_remove = string.punctuation
#                    table = {ord(char): None for char in chars_to_remove}
                    article = [removeNonWords(a) for a in article] 

                # if rnw:
                #     chars_to_remove = string.punctuation+'1234567890'
                #     table = {ord(char): None for char in chars_to_remove}
                #     article = article.translate(table)
                #     article = removeNonWords(article)
                #     article = removeStopWordsString(article)

                # temp
                articles.extend(article)
                for i in range(len(article)):
                    labels.append(isFake)
                    dates.append(row[dateCol])

                # articles.append(article)
                # labels.append(isFake)
                # dates.append(row[dateCol])

            except UnicodeError:
                print "non-unicode article"
            counter += 1

    return (articles, labels, dates)

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def removeNonWords(x):
    """
    Remove words that are not in the (English) dictionary
    @param string x
    @return string x with non-English words removed
    Example: 'The Unitedd States' --> 'The States'
    """
    word_list = list(x.split())
    filtered_words = [word for word in word_list if wordnet.synsets(word)]
    return ' '.join(filtered_words)

def removeNonWords2(x):
    """
    Remove words that are not in the (English) dictionary
    @param string x
    @return string x with non-English words removed
    Example: 'The Unitedd States' --> 'The States'
    """
    myString = list(x.split())
    wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

    filtered_words = [word for word in myString if word in wordlist]
    return ' '.join(filtered_words)


def removeStopWordsString(x):
    """
    Remove stop words from a string.
    @param string x
    @return string x with stop words removed
    Example: 'The United States' --> {'United': 1, 'States': 1}
    """
    stop_words = set(stopwords.words('english'))
    l = list(x.split())
    
    for stop_word in stop_words:
        l = [word for word in l if word != stop_word]
    
    return ' '.join(l)