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



def main():

    # set seed
    np.random.seed(1)

    # load up the data
    nyt, nytLabels, nytDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/NYT.csv", 
        -1, 3, 5)
    cnn, cnnLabels, cnnDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/CNN.csv", 
        -1, 3, 5)
    usa, usaLabels, usaDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/USA.csv", 
        -1, 0, 4)
    lat, latLabels, latDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/LAT.csv", 
        -1, 3, 5)
    wapo, wapoLabels, wapoDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/WPO.csv", 
        -1, 3, 5)
    fake, fakeLabels, fakeDates = util.loadArticles("../../Datasets/Datasets -- Cleaned/fake-excel-cleaned-3.csv", 
        1, 5, 3)

    data = fake + nyt + cnn + usa + lat + wapo
    labels = fakeLabels + nytLabels + cnnLabels + usaLabels + latLabels + wapoLabels
    dates = fakeDates + nytDates + cnnDates + usaDates + latDates + wapoDates

    print len(dates)
    print len(fakeDates)

    # sort by date
    allData = zip(data, labels, dates)
    allData.sort(key = lambda x:datetime.strptime(x[2], '%m/%d/%y'))
    data = [x[0] for x in allData]
    labels = [x[1] for x in allData]
    dates = [x[2] for x in allData]

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

        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']

        util.show_most_informative_features(vectorizer, classifier, n = 50)
        t()








# This makes the interpreter load up all functions and then start the main() function
if __name__=="__main__":
     main()


#     # Load the speeches from the data directory
#     data = util.loadSpeeches('../data', True, False)
#     print '----------------------------------------'
#     examples, presidents = util.convertDataset(data, 1)
    
#     # Pre-proces the transcripts
#     rawCorpus, allLabels = zip(*examples)
#     allCorpus = pr.preProcess(rawCorpus, English=True)
#     cleanExamples = zip(allCorpus,allLabels)

#     # Define list of vectorizers
#     vectorizer_list = [CountVectorizer(ngram_range=[2,2], binary = False), \
#                        CountVectorizer(ngram_range=[1,1], binary = True), \
#                        CountVectorizer(ngram_range=[1,2], binary = True), \
#                        CountVectorizer(ngram_range=[2,2], binary = True),
#                        CountVectorizer(ngram_range=[1,1], binary = False), \
#                        CountVectorizer(ngram_range=[1,2], binary = False)]

#     # Define list of classifiers
#     classifier_list = [QDA(priors = [0.5, 0.5])]

#     # Loop over all possible methods
#     meanErrors = []

#     print '----------------------------------------'
#     print 'Looping over all methods...'
#     print '----------------------------------------'
    
#     for vectorizer,classifier in itertools.product(vectorizer_list, classifier_list):
#         print 'Vectorizer: grams with range ', vectorizer.ngram_range
#         print 'Classifier: ', classifier.__class__.__name__

#         # Define pipeline
#         textcf = Pipeline([
#             ('vectorizer', vectorizer),
#             ('tfidf', TfidfTransformer()),
#             ('to_dense', DenseTransformer()),
#             ('classifier', classifier),
#         ])
    
#         val_errors = validation_errors(cleanExamples, presidents, textcf, verbose=False)
#         meanErrors += [np.mean(val_errors)]
#         print 'Average test error: %f\n'%(np.mean(val_errors))
#         print '----------------------------------------'

#     print '----------------------------------------'
#     print 'SUMMARY'
#     print '----------------------------------------'
    
#     i=0
#     print 'Vectorizer \t %30s \t\t Test error'%('Classifier')
#     for vectorizer,classifier in itertools.product(vectorizer_list, classifier_list):
#         print 'grams range [%d,%d], \t %30s \t %f' \
#             %(vectorizer.ngram_range[0], vectorizer.ngram_range[1], classifier.__class__.__name__, \
#               meanErrors[i])
#         i+=1

# # Compute hold-out prediction error rates
# def validation_errors(examples, presidents, textcf, verbose=True):
#     meanScore = []
    
#     # Choose cross-validatation scheme (switch to util.cvByPresident(presidents) to CV by pairs)
#     # cvFolds = util.cvBySpeech(presidents)
#     cvFolds = util.cvByIndividualPresident(presidents)
#     # cvFolds = KFold(shuffle=True,n_splits=10).split(range(len(examples)))

#     # Train and validate
#     for train,test in cvFolds:
#         # Training
#         trainExamples = [examples[i] for i in train]
#         trainCorpus, trainLabels = zip(*trainExamples)
#         model = textcf.fit(trainCorpus, trainLabels)
#         t()


#         # comment out line below to turn off tree-printing 
#         # util.printTree(model.named_steps['classifier'], 
#         #                model.named_steps['vectorizer'])
            
#         # Evaluation on hold-out data
#         testExamples = [examples[i] for i in test]
#         testCorpus, testLabels = zip(*testExamples)
#         predicted = textcf.predict(testCorpus)
#         meanScore += [np.mean(predicted != testLabels)]

#         if verbose:
#             # Print most informative features
#             vectorizer = model.named_steps['vectorizer']
#             print txtlearn.show_most_informative_features(model, text=None, n=50)

#             # Show training/validation split and predictions on test data
#             print '--------------------'
#             print set([presidents[p] for p in train])
#             print set([presidents[p] for p in test])
#             for x in zip(testLabels, predicted): print x
#             print 'Test error: %f\n'%(np.mean(predicted != testLabels))
#             print '--------------------'

#     return meanScore


# # This makes the interpreter load up all functions and then start the main() function
# if __name__=="__main__":
#     main()
