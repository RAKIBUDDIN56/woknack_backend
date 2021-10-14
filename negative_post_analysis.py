import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import re
import string
import nltk
import warnings
# %matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings('ignore')

def nagative_post_analysis(post):
    #Loading the dataset
    df = pd.read_csv(r'C:\Users\user\Desktop\RP\PP2\Datasets\train.csv\train.csv')
    # print(df.head())
    # datatype info
    # df.info()
    #Preprocessing the dataset
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for word in r:
            input_txt = re.sub(word, "", input_txt)
        return input_txt
    # df.head()
    # remove twitter handles (@user)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    # df.head()
    # remove special characters, numbers and punctuations
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
    # df.head()
    # remove short words
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
    # df.head()
    # individual words considered as tokens
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    tokenized_tweet.head()
    # stem the words
    
    stemmer = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    tokenized_tweet.head()
    # combine words into single sentence
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
    df['clean_tweet'] = tokenized_tweet
    # df.head()
    # feature extraction

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # newly added
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5001)

    bow = bow_vectorizer.fit_transform(df['clean_tweet'])

   
    x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.25)
    #Model Training
    # training
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # pred = model.predict(x_test)
    post  =[post]
    XX = bow_vectorizer.transform(post)
    yy=model.predict(XX)
    
    for x in yy:
        print('Mode is :',x)
        if x == 0:
            result = 'Positive'
        elif x == 1:
            result = 'Negative'


    return result

