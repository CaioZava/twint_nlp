import re
import tqdm
import twint
import spacy
import gensim
import pyLDAvis
import numpy as np
import pandas as pd
#import contractions
#import seaborn as sns
from pprint import pprint
from datetime import datetime
#import matplotlib.pyplot as plt
import gensim.corpora as corpora
from nltk.corpus import stopwords
from itertools import combinations
from gensim.models import CoherenceModel
from langid.langid import LanguageIdentifier, model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

pd.options.display.max_colwidth = 600
pd.options.display.max_columns=500


def get_tweets(since = '2021-07-19 17:00:00', until = '2021-07-19 17:05:00', keywords = '#covid'):
    c = twint.Config()
    c.Search = keywords
    c.Limit = None
    c.Since = since
    c.Until = min(datetime.strptime(until, "%Y-%m-%d %H:%M:%S"), datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    c.Pandas = True
    c.Hide_output = True

    # Run
    twint.run.Search(c)
    df1 = twint.storage.panda.Tweets_df
    df1 = df1[['id', 'created_at', 'date', 'tweet', 'user_id', 'user_id_str', 'username', 'nlikes', 'nretweets']]
    return df1


def remove_links(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'bit.ly/\S+', '', tweet)
    tweet = tweet.strip('[link]')
    return tweet


def remove_users(tweet):
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    return tweet


def reduce_lengthening(word):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", word)


def clean_tweets(tweet, my_punctuation='!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('[' + my_punctuation + ']+', ' ', tweet)
    tweet = re.sub('\s+', ' ', tweet)
    tweet = re.sub('([0-9]+)', '', tweet)
    return tweet


def filter_df_by_lang(df, col='tweet',
                      lang='en'):  # funcao que filtra linhas de df onde a coluna col nao esta na lingua lang
    langs = df[col].apply(identifier.classify)
    explode = langs.explode()
    return (df.loc[explode[~explode.index.duplicated(keep='first')] == lang])


def sent_to_words(list_tweets):
    for tweet in list_tweets:
        yield(gensim.utils.simple_preprocess(str(tweet)))

df = get_tweets(since = '2021-07-19 17:00:00', until = '2021-07-19 18:00:00', keywords = '#covid')
df = filter_df_by_lang(df, col = 'tweet', lang = 'en').reset_index(drop = True)
df['tweet'] = df['tweet'].apply(clean_tweets)
data = df.tweet.values.tolist()
data_words = list(sent_to_words(data))

stop_words = stopwords.words('english')
stop_words.extend(['tweet'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            #print(str(pair[0]) + " " + str(pair[1]))
            pair_scores.append( w2v_model.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)


class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok.lower() in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 2:
                    tokens.append( tok.lower() )
            yield tokens

## Creating document term matrix using TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english',ngram_range=(1,1))#(max_df=0.95, min_df=3, stop_words='english',ngram_range=(1,2))
tweets_clean = [' '.join(i) for i in data_words_nostops]
dtm = tfidf.fit_transform(tweets_clean)

docs_raw = tweets_clean
docgen = TokenGenerator(docs_raw, stop_words)
w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=3, sg=1)

# creating NMF model with 6 components

nmf_model = NMF(n_components=6, random_state=420)

# fitting and transforming dtm obtained,
# to get weights corresponding to belongingness of the document to each topic

topics=nmf_model.fit_transform(dtm)


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


df['Topic']=topics.argmax(axis=1)

## Naming topics

naming={0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g'}
df['Topic_name']=df['Topic'].map(naming)


df[['OriginalTweet','Topic_name']].head()