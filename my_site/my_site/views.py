from pathlib import Path
import joblib
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

from nltk.corpus import stopwords as nltk_stopwords

BASE_DIR = Path(__file__).resolve().parent.parent
model_class = joblib.load(BASE_DIR / "my_site/templates/model_clas.joblib")
model_reg = joblib.load(BASE_DIR / "my_site/templates/model_reg.joblib")
vocabulary = joblib.load(BASE_DIR / "my_site/templates/vocabulary.json")
 
def index(request):
    text = request.POST.get("feedback")
    if text:
        # Лемматизация текста
        text = text.lower()
        # Удаление всех символов, кроме английских букв и пробелов
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # Удаление лишних пробелов
        text = " ".join(text.split())


        text = [text]

        count_tf_idf = TfidfVectorizer(vocabulary=vocabulary)
        text = count_tf_idf.fit_transform(text)
        pred = model_class.predict(text)
        pred_reg = model_reg.predict(text)

        if pred[0] == 0:
            result1 = 'Отрицательный отзыв'
        else:
            result1 = 'Положительный отзыв'

        if pred_reg[0] < 1:
            result2 = 1
        elif pred_reg[0] > 10:
            result2 = 10
        else:
            result2 = round(pred_reg[0])

        try:
            return render(request, 'index.html', {'feedback_status': result1, 'rating': result2})
            # return HttpResponse(f"<h2>{result1, result2}")

        except Exception as e:
            logger.error(f"Error in my_view: {str(e)}")
            return HttpResponseServerError("Internal Server Error")
        
    else:
        return render(request, "index.html")
 

