from flask import Flask,request, jsonify, render_template
import nltk 
from nltk.tokenize import word_tokenize
import re 
import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

posts=pd.read_csv('C:\\Users\\DELL\\Desktop\\facebook_posrs.csv', encoding = "cp1252")

def simple_split(data,y,length,split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n = int(split_mark*length)
    else:
        n = int(split_mark)
    X_train = data [:n].copy()
    X_test = data[n:].copy()
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return X_train,X_test,y_train,y_test

vect = CountVectorizer()

X_train,X_test,y_train,y_test = simple_split(posts.message,posts.label,len(posts))

X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if (request.method=='POST'):
        message = request.form['message']
        operation=request.form['operation']
        
        if(operation=='Logistic Regression'):
            predic1 = (logreg.predict(vect.transform([message]))[0])
            output = 'LogiticRegression = {}'.format(predic1)
        
        if(operation=='Naive Baise'):
            predic2 = (nb.predict(vect.transform([message]))[0])
            output = 'Naive Baise = {}'.format(predic2)
        
        if(operation=='Random Forest'):
            predic3 = (rf.predict(vect.transform([message]))[0])
            output = 'Random Forest = {}'.format(predic3)
        
        
    
    return render_template('index.html', result ='Output is {}'.format(output))
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
