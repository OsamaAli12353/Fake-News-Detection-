import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import re
import seaborn as sns
import sklearn.feature_extraction.text as text
from nltk.stem import SnowballStemmer
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/kamal/Downloads/NLProject/NLProject/news.csv', usecols=['text', 'label'])
data['class'] = np.where(data['label'] == 'FAKE', 0, 1)
data.drop_duplicates(inplace=True)

contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not"}
# Regular expression for finding contractions
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


# Expanding Contractions in the News
data['text'] = data['text'].apply(lambda x: expand_contractions(x))

data['text'] = data['text'].str.lower()

# remove punctuation

data['text'] = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

# remove words and digits

data['text'] = data['text'].apply(lambda x: re.sub('W*dw*', '', x))
data['text'] = (
    data['text'].
    str.replace('[^A-Za-z0-9\s]', '', regex=True).
    str.replace('\n', '', regex=True).
    str.replace('\s+', ' ', regex=True)
)

# remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


data['text'] = data['text'].apply(lambda x: remove_stopwords(x))

# rephrase text url

data['text'] = data['text'].apply(lambda x: re.sub('(http[s]?S+)|(w+.[A-Za-z]{2,4}S*)', 'urladd', x))

# stemming
stemmer = PorterStemmer()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


data["text"] = data["text"].apply(lambda x: stem_words(x))

#Lemmtization

lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


data["text"] = data["text"].apply(lambda text: lemmatize_words(text))

# remvoe extra spaces
data["text"] = data["text"].apply(lambda text: re.sub(' +', ' ', text))

data["text"][0]

# save data clean after preprocssiong
data.to_csv('C:/Users/kamal/Downloads/NLProject/NLProject/news_clean.csv', index=False)

#visualization

text = ' '.join(data['text'].tolist())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)


# Create a bar chart of the most common words in the dataset
word_counts = Counter(text.split())
top_words = word_counts.most_common(20)

plt.figure(figsize=(16, 12))
sns.barplot(x=[w[0] for w in top_words], y=[w[1] for w in top_words])
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.show()

# Create a scatter plot of the word counts for each article
article_word_counts = [len(article.split()) for article in data['text'].tolist()]

plt.figure(figsize=(16, 12))
sns.scatterplot(x=range(len(article_word_counts)), y=article_word_counts)
plt.title('Word Counts for Each Article')
plt.xlabel('Article Number')
plt.ylabel('Word Count')
plt.show()
