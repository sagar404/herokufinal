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
posts

#It will give you the split of total negative(1) and total postive(0) label
print("Samples per class: {}".format(np.bincount(posts.label)))

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

#Train and Test Split Counts
X_train,X_test,y_train,y_test = simple_split(posts.message,posts.label,len(posts))
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#Positive and Negative count of label in Train and Test
print("Samples per class: {}".format(np.bincount(y_train)))
print("Samples per class: {}".format(np.bincount(y_test)))


#Learn all the vocabulary words from X_train and apply tranformation to build the bag of words on X_test as well as transform the Train
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

feature_names = vect.get_feature_names()
print ("Number of feature: {}".format(len(feature_names)))
print ("First 20 features:\n{}".format(feature_names[:20]))
print ("Features 300 to 320:\n{}".format(feature_names[300:320]))
print ("Every 100th feature:\n{}".format(feature_names[::100]))

#Succsesfuly we have create the vocabulary and now print the vocabulary
vect.vocabulary_

#Implementation of LogisticRegression
#Calculate the expected accuracy of the model  
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

#Confussion Matrix used to describe the Performance of the classification 
pred_logreg = logreg.predict(X_test)
confusion = confusion_matrix(y_test, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))

#Classification Rate or Accuracy
accuracy = accuracy_score(y_test, pred_logreg) 
print("Accuracy: {:.2f}".format(accuracy))

#Implementation of NaiveBayse
nb = MultinomialNB()
nb.fit(X_train, y_train)
print("Training set score: {:.3f}".format(nb.score(X_train, y_train)))
print("Test set score: {:.3f}".format(nb.score(X_test, y_test)))

#Confussion Matrix used to describe the Performance of the classification 
pred_nb = nb.predict(X_test)
confusion = confusion_matrix(y_test, pred_nb)
print("Confusion matrix:\n{}".format(confusion))

#Classification Rate or Accuracy
accuracy = accuracy_score(y_test, pred_nb) 
print("Accuracy: {:.2f}".format(accuracy))

#Implementaion of RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Testining set score: {:.3f}".format(rf.score(X_train, y_train)))
print("Test set score: {:.3f}".format(rf.score(X_test, y_test)))

#Testing the prediction
#message = "I am depressed"
#print("Logreg Prediction")
#print(logreg.predict(vect.transform([message]))[0])
#print("RF Prediction")
#print(rf.predict(vect.transform([message]))[0])
#print("NB Prediction")
#print(nb.predict(vect.transform([message]))[0])
