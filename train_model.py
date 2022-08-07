from numpy import vectorize
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
import fitz
import pickle

vectorizer = CountVectorizer()

def pre_process_df():
    f_df = pd.DataFrame(columns=['Text','labels'])
    df = pd.read_csv("tabular_dataset.csv")
    f_df['Text'] = df["text"]
    f_df['labels'] = df["labels"]
    return f_df

def input_process(text):
    translator = str.maketrans('','',string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)

def remove_stop_words(input):   
    final_words = []
    for line in input:
        line  = input_process(line)
        final_words.append(line)
    return final_words

def train_model(df):
    input = df['Text']
    output = df['labels']
    input = remove_stop_words(input)
    df['Text'] = input
    input = vectorizer.fit_transform(input)
    nb = MultinomialNB()
    nb.fit(input,output)
    return nb
    
if __name__ == "__main__":
    df = pre_process_df()
    model = train_model(df)
    pickle.dump(model,open('classifier.model','wb'))
    pickle.dump(vectorizer,open('vecorizer.pickle','wb'))
    