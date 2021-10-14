from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
import numpy as np

def category_prediction(post):
    # train_csv = pd.read_csv(r'C:\Users\user\Desktop\RP\Dataset\py files\all_topic.csv') # path to file
    train_csv = pd.read_csv(r'.\all_topic.csv') # path to file
    train_csv_new=train_csv.dropna()
    train_X = train_csv_new['Processed_post']   
    train_y = train_csv_new['category']
    # test_csv = pd.read_csv(r'C:\Users\user\Desktop\RP\PP1\all_topic_test.csv') # path to file
    test_csv = pd.read_csv(r'.\all_topic_test.csv') 
    #remove the NaN object
    test_csv_new =test_csv.dropna()
    test_X = test_csv_new['Processed_post']
    test_y = test_csv_new['category']

    #t = time()  # not compulsory

    # loading TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5001)

    tfidf = tfidf_vectorizer.fit_transform(train_X.values.astype('U'))

    #duration = time() - t

    #print("Time taken to extract features from training data : %f seconds" % (duration))

    #print("n_samples: %d, n_features: %d" % tfidf.shape)

    #t = time()

    X_test_tfidf = tfidf_vectorizer.transform(test_X.values.astype('U'))
    #re = tfidf_vectorizer.transform(category.values.astype('U'))


    # duration = time() - t
    # print("Time taken to extract features from test data : %f seconds" % (duration))
    # print("n_samples: %d, n_features: %d" % X_test_tfidf.shape)

    #Logistic Regression classifier
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(tfidf, train_y)
    # compute the performance measures
    #y_pred=clf.predict(X_test_tfidf)
    #score1 = metrics.accuracy_score(y_pred,test_y)
    #print("Accuracy is :   %0.3f\n" % score1)
    # X = np.array(post, dtype=float)
    # Y= X.reshape((1,-1))
    post  =[post]
    XX = tfidf_vectorizer.transform(post)
    yy=clf.predict(XX)
    for x in yy:
        print('Category is :',x)

    return  clf.predict(XX)




    
     

