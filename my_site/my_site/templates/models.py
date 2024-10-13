from django.db import models
import joblib
import nltk
import re
import spacy
nltk.download('stopwords')
#from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords as nltk_stopwords

model_class = joblib.load('model_clas.joblib')
count_tf_idf = joblib.load('count_tf_idf.joblib')


# найдём стоп-слова (слова без смысловой нагрузки)
#stopwords = list(nltk_stopwords.words('english'))
# создадим счётчик и передадим в него список стоп-слов
#count_tf_idf = TfidfVectorizer(stop_words=stopwords)
nlp = spacy.load("en_core_web_sm")

def predict_class(input_data): 
    text = input_data
    
    # Лемматизация текста
    text = text.lower()
    # Удаление всех символов, кроме английских букв и пробелов
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Удаление лишних пробелов
    text = " ".join(text.split())
    text = nlp(text)


    text = [text]
    text = count_tf_idf.transform(text)
    pred = model_class.predict(text)
    if pred[0] == 0:
        return 'Отрицательный отзыв'
    else: 
        return 'Положительный отзыв'

