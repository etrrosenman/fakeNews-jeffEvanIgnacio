import numpy as np
import nltk, sklearn, random, operator, itertools, inspect
import util

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
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from utils.dataset import DataSet
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize



def load_and_preprocess_data(debug = False):
    dataset = DataSet() 
    stances = dataset.stances
    articles = dataset.articles
    if debug:
        allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance'], s['Body ID']) for s in stances[:200]]
    else:
        allData = [(s['Headline'].split(), articles[s['Body ID']].split(), s['Stance'], s['Body ID']) for s in stances]
    
    # choose specific articles to comprise the training set
    articleList = articles.keys()
    np.random.seed(2017)
    np.random.shuffle(articleList)

    trainArticleIndices = articleList[:int(len(articleList)*4/5)]
    devArticleIndices = articleList[int(len(articleList)*4/5):]

    train = [v[:3] for v in allData if v[3] in trainArticleIndices]
    dev = [v[:3] for v in allData if v[3] in devArticleIndices]
    allData = [v[:3] for v in allData]


    helper = ModelHelper.build(allData)

    # now process all the input data.
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    return helper, train_data, dev_data, train, dev


def main():

    # set seed
    np.random.seed(1)

    # load up the data
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(debug)

    # Define list of vectorizers
    vectorizer_list = [CountVectorizer(ngram_range=[1,1], binary = False)]

    # define list of classifiers 
    classifier_list = [LogisticRegression(penalty='l1', C=10000.0), 
                       MultinomialNB()]

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

    # iterate through the list
    for vectorizer, classifier in itertools.product(vectorizer_list, classifier_list):
        print 'Vectorizer: grams with range ', vectorizer.ngram_range
        print 'Classifier: ', classifier.__class__.__name__

        # Define pipeline
        textcf = Pipeline([
            ('vectorizer', vectorizer),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier),
        ])

        # fit the model
        model = textcf.fit(trainData, trainLabels)
        fitted = model.predict(testData)
        print(np.sum(fitted != testLabels)/float(len(fitted)))



# This makes the interpreter load up all functions and then start the main() function
if __name__=="__main__":
     main()

