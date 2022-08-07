
from numpy import vectorize
import sklearn
import nltk
import pickle
import fitz
from sklearn.feature_extraction.text import CountVectorizer


def load_model_and_vectorizer():
    model = pickle.load(open("classifier.model",'rb'))
    vectorizer = pickle.load(open('vecorizer.pickle','rb'))
    return model,vectorizer

if __name__ == "__main__":
    model,vectorizer = load_model_and_vectorizer()
    path = input("Enter path of the file: ")
    doc = fitz.open(path)
    content = ''
    for page in range(len(doc)):
        content = content+doc[page].get_text()
    
    content = vectorizer.transform([content]) 
    pred = model.predict(content)
    if pred[0] == 1:
        print('This is document about AI')
    else:
        print('This is document about Web')
              