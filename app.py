from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

books_small = pickle.load(open('books_small.pkl','rb'))
# books_bow = pickle.load(open('books_bow.pkl','rb'))
# countvector = pickle.load(open('countvector.pkl','rb'))

countvector = CountVectorizer(ngram_range=(1,1))
books_bow = countvector.fit_transform(books_small['final'])

tfidf = TfidfVectorizer(ngram_range=(1,1))
books_tfidf = tfidf.fit_transform(books_small['final'])

stop = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',    'yourself', 'more', 'if', 'on', 'don', "don't", 'once', 'this', 'being', 'as', 'there', 'should', 'its', 'been', 'didn', "you're", 'me', 'but', 'than', 'just', 'an', 'when', 'after', 'now', 'your', "needn't", 'aren', 'that', 'during', 've', 'yourselves', "you'll", 'who', 'whom', "hadn't", "weren't", "shouldn't", 'at', 'the', 'can', 'herself', 'from', 'those', "she's", 'hers', 'so', 'for', 'll', 'do', "didn't", 't', 's', 'to', 'wouldn', "hasn't", 'before', "shan't", 'too', "mightn't", 'above', 'most', 'him', 'theirs', 'has', 'she', 'i', 'here', 'be', 'because', 'd', 'y', "couldn't", 'doesn', "wouldn't", 'ma', 'all', 'doing', 'himself', 'are', 'o', "doesn't", 'what', 'my', 'up', "you've", 'nor', 'couldn', 'ours', 'his', 'themselves', 'which', 'in', 'having', "isn't", 'while', 'shan', 'below', 'ain', "won't", 'these', 'needn', "it's", 'why', 'were', 'mightn', 'won', 'itself', 'mustn', 'was', 'against', 'of', 'then', 'both', "wasn't", 'by', 'or', 'myself', 'a', 'only', 'we', 'down', 'no', 'between', 'some', 'hadn', 'where', 'until', 'other', 'did', 'they', 'have', "haven't", 'further', 'you', 'had', 'yours', 'through', 'same', "should've", 'he', 'off', 'will', 'few', 'ourselves', 'how', "aren't", 'wasn', 'our', 'isn', 'them', 'into', "that'll", "mustn't", 'such', 'their', 'hasn', 'm', 'weren', 'about', 'under', 'it', 'does', 'not', 'haven', "you'd", 'over', 'out', 'any', 'and', 'each', 'very', 'her', 'with', 'own', 're', 'shouldn', 'am', 'again', 'is']
lemmatizer = WordNetLemmatizer()

def preprocess(user_text):
    out= []
    user_text = [user_text]
    for i in user_text:
        ind = ' '.join(re.findall(r'\w+|\d+',i))
        ind = ind.lower()
        ind = word_tokenize(ind)
        ind = [lemmatizer.lemmatize(word) for word in ind if word not in stop]
        out.append(" ".join(ind))
    return out

def recommend(user_text):
    ind = preprocess(user_text)
    ind1 = preprocess(user_text)
    x = pd.Series(cosine_similarity(books_tfidf,countvector.transform(ind)).
               flatten()).nlargest(5).sort_values(ascending=False)
    ind= x.index.tolist()
    ind = books_small.loc[ind,['Title', 'description', 'authors', 'image', 'previewLink', 'publisher',
       'publishedDate', 'categories', 'ratingsCount']]
    ind['score'] = pd.Series(cosine_similarity(books_tfidf,tfidf.transform(ind1)).
               flatten()).nlargest(5).sort_values(ascending=False).values.tolist()
    return ind


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/recommend',methods = ["POST"])
def recommendation():

    user_text = request.form['user_text']
    ind = preprocess(user_text)
    x = pd.Series(cosine_similarity(books_tfidf,countvector.transform(ind)).
               flatten()).nlargest(5).sort_values(ascending=False)
    top_index = x.index.tolist()[0]
    
    top_score = pd.Series(cosine_similarity(books_tfidf,countvector.transform(ind)).
                          flatten()).nlargest(5).sort_values(ascending=False).values.tolist()[0]
    
    if top_score == 0:
        return render_template('warning.html',warning = 'No Match Found')  

        
    else:
        input_2 = books_small.loc[top_index,['final']][0]
        ind = recommend(input_2)

        titles = ind['Title'].values.tolist()
        authors = ind['authors'].values.tolist()
        images = ind['image'].values.tolist()
        links = ind['previewLink'].values.tolist()
        dates = ind['publishedDate'].values.tolist()
        cats = ind['categories'].values.tolist()
        publishers = ind['publisher'].values.tolist()

        return render_template('recommendations.html',
                            title1 = titles[0],title2 = titles[1],title3 = titles[2],title4 = titles[3],title5 = titles[4],
                            author1 = authors[0],author2 = authors[1],author3 = authors[2],author4 = authors[3],author5 = authors[4],      
                            image1 = images[0],image2 = images[1],image3 = images[2],image4 = images[3],image5 = images[4],
                            link1 = links[0],link2 = links[1],link3 = links[2],link4 = links[3],link5 = links[4],
                            date1 = dates[0],date2 = dates[1],date3 = dates[2],date4 = dates[3],date5 = dates[4],
                            cat1 = cats[0],cat2 = cats[1],cat3 = cats[2],cat4 = cats[3],cat5 = cats[4],
                            publisher1 = publishers[0],publisher2 = publishers[1],publisher3 = publishers[2],publisher4 = publishers[3],publisher5 = publishers[4],
                                )

if __name__ == '__main__':
    app.run(debug=True)