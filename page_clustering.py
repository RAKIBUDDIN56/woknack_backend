#imports libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# from wordcloud import WordCloud

def page_clustering():
    #import data from the given path
    data = pd.read_csv(r'C:\Users\user\Desktop\RP\PP2\Datasets\pages.csv')

    # print(data.head())
    Y = data[['Name', 'Category']]

    print(Y['Name'])

    categories = []
    names = []
    category =[]
    name = []
    with open(r'C:\Users\user\Desktop\RP\PP2\Datasets\pages.csv') as csvfobj: #csvfobj=object
        readCSV= csv.reader(csvfobj, delimiter=',') 
        for column in readCSV:
            category = column[2]
            name = column[1]
            categories.append(category)
            names.append(name)
# print(categories)
    # print(names)

    tfidfvect = TfidfVectorizer(stop_words='english')
    tfidf = tfidfvect.fit_transform(names)

    first_vector = tfidf[1]
 
    dataframe = pd.DataFrame(first_vector.T.todense(), index = tfidfvect.get_feature_names(), columns = ["tfidf"])
    dataframe.sort_values(by = ["tfidf"],ascending=False)
    # Sum_of_squared_distances = []
    # K = range(1,15)
    # for k in K:
    #     km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    #     km = km.fit(tfidf)
    #     Sum_of_squared_distances.append(km.inertia_)

    # plt.plot(K, Sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()
    num = 5
    model = KMeans(n_clusters = num, init = 'k-means++', max_iter = 200, n_init = 10)   
    model.fit(tfidf)
    print(model.cluster_centers_) 
    labels=model.labels_
    wiki_cl=pd.DataFrame(list(zip(categories,labels)),columns=['title','cluster'])
    # print(wiki_cl.sort_values(by=['cluster']))
    
    result={'cluster':labels,'name':names}
    result=pd.DataFrame(result)
    for k in range(0,num):
        s=result[result.cluster==k]
        text=s['name'].str.cat(sep=' ')
        text=text.lower()
        text=' '.join([word for word in text.split()])
        # wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(text)
        print('Cluster: {}'.format(k))
        print('Titles')
        titles=wiki_cl[wiki_cl.cluster==k]['title']
        print(titles.to_string(index=False))
        # plt.figure()
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis("off")
        # plt.show()
        X = tfidfvect.transform(["potion"])
    predicted = model.predict(X)
    print(predicted)


