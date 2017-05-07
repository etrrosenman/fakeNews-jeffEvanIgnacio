import numpy as np
import nltk, sklearn, random, operator, itertools, inspect
import util
import utils.glove as glove

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from pdb import set_trace as t
from datetime import date, time, datetime


def main():

    # set seed
    np.random.seed(1)

    # load up the data
    # nyt, nytLabels, nytDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/NYT.csv", 
    #     -1, 3, 5)
    cnn, cnnLabels, cnnDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/CNN.csv", 
        -1, 3, 5)
    # usa, usaLabels, usaDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/USA.csv", 
    #     -1, 0, 4)
    # lat, latLabels, latDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/LAT.csv", 
    #     -1, 3, 5)
    # wapo, wapoLabels, wapoDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/WPO.csv", 
    #     -1, 3, 5)
    # fake, fakeLabels, fakeDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/fake.csv", 
    #     1, 5, 3)

    # data = fake + nyt + cnn + usa + lat + wapo
    # labels = fakeLabels + nytLabels + cnnLabels + usaLabels + latLabels + wapoLabels
    # dates = fakeDates + nytDates + cnnDates + usaDates + latDates + wapoDates

    data = cnn
    labels = cnnLabels
    dates = cnnDates

    print len(dates)
#    print len(fakeDates)

    # sort by date
    allData = zip(data, labels, dates)
    allData.sort(key = lambda x:datetime.strptime(x[2], '%m/%d/%y'))
    data = [x[0] for x in allData]
    labels = [x[1] for x in allData]
    dates = [x[2] for x in allData]

    # defining training and test data -- use '11/16/16' as cutoff (gives 80-20 split) 
    indices = range(len(data))
    trainIndices = [i for i in indices if datetime.strptime(dates[i], '%m/%d/%y') < 
        datetime.strptime('11/16/16', '%m/%d/%y')]
    testIndices = [i for i in indices if datetime.strptime(dates[i], '%m/%d/%y') >= 
        datetime.strptime('11/16/16', '%m/%d/%y')]
    trainData = [data[i] for i in trainIndices]
    trainLabels = [labels[i] for i in trainIndices]
    testData = [data[i] for i in testIndices]
    testLabels = [labels[i] for i in testIndices]

    # load up the GloVe vectors
    unigramVectorizer = CountVectorizer(ngram_range=[1,1], binary = False)
    uni = unigramVectorizer.fit(data)
    wordList = uni.get_feature_names()
    wordDict = {} 
    counter = 0
    for u in wordList:
        wordDict[u] = counter 
        counter += 1

    wordVectors = glove.loadWordVectors(wordDict)

    # transform to glove
    t()
    


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))
    sentVector = np.mean([wordVectors[tokens.get(x)] for x in sentence], axis = 0)

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector



# This makes the interpreter load up all functions and then start the main() function
if __name__=="__main__":
     main()

