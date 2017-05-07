import csv
import string
from pdb import set_trace as t
import numpy as np
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.lancaster import LancasterStemmer

# up the size of the filed
csv.field_size_limit(10000000)

def loadArticles(csvPath, isFake, colNum, rnw = False):
    """
    Load all articles from a given file 
    """    

    articles = [] 
    labels = []

    with open(csvPath, 'rU') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None) # skip the header 

        counter = 0

        for row in reader:

            if counter % 20 == 0:
                print counter

            # kill all non-unicode articles (foreign language -- may want better 
            # way to deal with this in the future)
            try:
                article = row[colNum]
                article = article.decode('utf-8', 'ignore')
                if rnw:
                    chars_to_remove = string.punctuation+'1234567890'
                    table = {ord(char): None for char in chars_to_remove}
                    article = article.translate(table)
                    article = removeNonWords(article)
                    article = removeStopWordsString(article)

                articles.append(article)
                labels.append(isFake)

            except UnicodeError:
                print "non-unicode article"
            counter += 1

    return (articles, labels)

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

def clean(x):
    """
    Remove words that are not in the (English) dictionary
    @param string x
    @return string x with non-English words removed
    Example: 'The Unitedd States' --> 'The States'
    """

    # get a dictionary
    wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

    # get a stemmer
    stemmer = LancasterStemmer()

    # removal chars
    chars_to_remove = string.punctuation+'1234567890'
    table = {ord(char): None for char in chars_to_remove}
    myString = list(x.split())

    filtered_words = [word for word in myString if \
        wordnet.synsets(word.lower().translate(table)) or \
        word.lower().translate(table) in wordlist]
    return ' '.join(filtered_words)
