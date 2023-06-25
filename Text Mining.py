# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:54:04 2023

@author: madug
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import string
import spacy
from matplotlib.pyplot import imread
from wordcloud import WordCloud
%matplotlib inline



df=pd.read_csv('C:/Users/madug/Downloads/Elon_musk.csv',encoding='latin1')
df

#EDA Analysis

df.shape
df.isnull().sum()
df.dtypes
df1 = df.copy()
df.drop(['Unnamed: 0'],inplace=True,axis=1)
df

#Text Preprocessing

df=[Text.strip() for Text in df.Text]
df=[Text for Text in df if Text]
df[0:10]

# Joining the list into one string 
text=' '.join(df)
text

#Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation))
no_punc_text

#Tokenization

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_punc_text)
text_tokens 
len(text_tokens)

#Removing Stopwords with help of nltk
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')
sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)
no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)

# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])

#Stemming
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])
len(stemmed_tokens)

#Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)

lemmas=[token.lemma_ for token in doc]
print(lemmas)

clean_tweets=' '.join(lemmas)
clean_tweets

#Feature Extaction
1.Using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(lemmas)
print(cv.vocabulary_)

import warnings
warnings.filterwarnings('ignore')
print(cv.get_feature_names()[100:200])
print(x.toarray())
print(x.toarray().shape)


#Feature Extraction
Using Stemming

x1 = cv.fit_transform(stemmed_tokens)
x1

x1.toarray().shape

print(cv.vocabulary_)






2. Count Vectorizer with N-grams (Bigrams & Trigrams)

cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)
print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())




3.TF-IDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())

Generate Word Cloud

# Define a function to plot word cloud 

def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')

# Generate Word Cloud

from nltk.corpus import stopwords
wordcloud=WordCloud(width=3000, height=2000, background_color='black', max_words=50, colormap='Set1').generate(text)
plot_cloud(wordcloud)

Named Entity Recognition (NER)

# Parts Of Speech  Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)
one_block

for token in doc_block[100:200]:
    print(token,token.pos_)
    
# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])    

# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)
words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)
wf_df=pd.DataFrame(words_freq)
wf_df.columns=['word','count']
wf_df[0:10] # viewing top ten results

# Visualizing results 
#Barchart for top 10 nouns + verbs
wf_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs')




Emotion Mining - Sentiment Analysis

from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(df))
sentences

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df
wf_df

scores = wf_df.set_index('word')['count'].to_dict() 
scores

sentiment_lexicon = scores
def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score 
#test that it works
calculate_sentiment(text = 'amazing')

sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment) 
sent_df

sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10) 

sent_df.sort_values(by='sentiment_value').tail(10) 

# Sentiment score of the whole review
sent_df['sentiment_value'].describe() 

# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0].head() 
sent_df[sent_df['sentiment_value']>=20].head() 

sent_df['index']=range(0,len(sent_df)) 
import seaborn as sns
sns.distplot(sent_df['sentiment_value'])
plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df) 

sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')






