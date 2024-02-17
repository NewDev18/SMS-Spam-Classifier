import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("E-MAIL/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
           y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Check'):
    transformed_msg = transform_text(input_sms)
    vectorized_msg = tfidf.transform([transformed_msg])
    result = model.predict(vectorized_msg)

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
