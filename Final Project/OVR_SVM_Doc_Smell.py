# Importing necessary libs
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
from sklearn.multiclass import OneVsRestClassifier
import pickle
import glob
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
from sklearn.metrics import confusion_matrix

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# defining necessary function for data pre-process and result generation
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def remove_newline_chars(sentence):
	sentence = sentence.replace('\n'," ")
	sentence = sentence.replace('\r'," ")
	sentence = sentence.replace('\t'," ")
	return sentence



def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])








### loading new data for prediction

all_files = glob.glob("all labelled sample sets-extension/" + "/*.xlsx")
new_data = pd.DataFrame()

for filename in all_files:
    df = pd.read_excel(filename)
    try:
      df['Documentation Text'] = df['Documentation Text'].apply(clean_text)
    except:
      print(filename)
      pass
    new_data = new_data.append(df,ignore_index=True)


new_data = new_data.dropna(axis=1, how='any') #deleting (auto-generated) extra columns (containing all NaN values



original_documentation_text_list = new_data['Documentation Text'].tolist()

print("data load done")




# data format and pre-process

categories = ['Fragmented','Tangled','Excessive Structured','Bloated','Lazy']
number_of_class = len(categories)
print(number_of_class)


documentation_text_without_method_prototype_list = []

for index,row in new_data.iterrows():
	documentation_text = row['Documentation Text']
	method_prototype = row['Method Prototype']
	documentation_text_without_method_prototype = documentation_text.replace(method_prototype," ")
	documentation_text_without_method_prototype_list.append(documentation_text_without_method_prototype)


new_data['Documentation Text'] = documentation_text_without_method_prototype_list
new_data['Documentation Text'] = new_data['Documentation Text'].apply(remove_newline_chars)
new_data['Documentation Text'] = new_data['Documentation Text'].str.lower()
new_data['Documentation Text'] = new_data['Documentation Text'].apply(removeStopWords)
new_data['Documentation Text'] = new_data['Documentation Text'].apply(stemming)


print("data clean and format done")




x_new = new_data[['Documentation Text']].copy()
x_new = x_new['Documentation Text'].tolist()



## Evaluation

print("using SVM with ovr:")
print("------------------------------")
print("")

# loading model
filename = 'ovr_svm_bow_saner.sav'
classifier = pickle.load(open(filename, 'rb'))



tfidf = CountVectorizer()
tfidf.fit(x_train)
x_train = tfidf.transform(x_train)
x_new = tfidf.transform(x_new)

print("data transform done")

predictions = classifier.predict(x_new)


pred_label_df = pd.DataFrame(predictions, columns=categories)
test_label_df = new_data[categories]


# Result generation

for category in categories:
  this_category_test_list = test_label_df[category].tolist()
  this_category_pred_list = pred_label_df[category].tolist()
  acc_for_this_category = accuracy_score(this_category_test_list,this_category_pred_list)
  precision_for_this_category = precision_score(this_category_test_list,this_category_pred_list)
  recall_for_this_category = recall_score(this_category_test_list,this_category_pred_list)
  f1_for_this_category = f1_score(this_category_test_list,this_category_pred_list)
  print('result for '+category+' smell type:')
  print('accuracy: '+ str(acc_for_this_category))
  print('precision: '+ str(precision_for_this_category))
  print('recall: '+ str(recall_for_this_category))
  print('f1 score: '+ str(f1_for_this_category))
  print_confusion_matrix(this_category_test_list,this_category_pred_list)
  print('-------------------------------------------------------------')
  print('')