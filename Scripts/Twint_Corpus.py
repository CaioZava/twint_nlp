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


def get_tweets(since = '2021-07-19 17:00:00', until = '2021-07-20 17:00:00', keywords = '#covid'):
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

df = get_tweets(since = '2021-07-19 17:00:00', until = '2021-07-20 17:00:00', keywords = '#covid')
df = filter_df_by_lang(df, col = 'tweet', lang = 'en').reset_index(drop = True)
df['tweet'] = df['tweet'].apply(clean_tweets)
data = df.tweet.values.tolist()
data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

stop_words = stopwords.words('english')
stop_words.extend(['tweet'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
#data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_uci')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

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


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b, w2v_model):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    term_rankings = [None]*k
    for topic_id in range(k):
        term_rankings[topic_id] =  [i for i,j in lda_model.show_topic(topic_id)]
    coherence_model_lda = calculate_coherence(w2v_model, term_rankings)

    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_uci')

    return coherence_model_lda#.get_coherence()




grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
    # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
    #gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
    corpus]
corpus_title = ['100% Corpus']#'75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }

tweets_clean = [' '.join(i) for i in data_lemmatized]
docs_raw = tweets_clean

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

docgen = TokenGenerator(docs_raw, stop_words)
w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=1, sg=1)
#'column' in list(w2v_model.wv.vocab.keys())
#'column' in  [i for j in data_words_nostops for i in j]
#'column' in  [i for j in data_words for i in j]
#'column' in  [i for j in term_rankings for i in j]

# Can take a long time to run

if 1 == 1:
    pbar = tqdm.tqdm(total=(len(corpus_sets)*len(topics_range)*len(alpha)*len(beta)))

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                  k=k, a=a, b=b, w2v_model=w2v_model)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()


########################################################################################################################
## Creating document term matrix using TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english',ngram_range=(1,1))#(max_df=0.95, min_df=3, stop_words='english',ngram_range=(1,2))
tweets_clean = [' '.join(i) for i in data_words_nostops]
dtm = tfidf.fit_transform(tweets_clean)

docs_raw = tweets_clean
docgen = TokenGenerator(docs_raw, stop_words)
w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=3, sg=1)

kmin, kmax = 2, 10

topic_models = []
# try each value of k
for k in range(kmin,kmax+1):
    print("Applying NMF for k=%d ..." % k )
    # run NMF
    model = NMF( init="nndsvd", n_components=k )
    W = model.fit_transform(dtm)
    H = model.components_
    # store for later
    topic_models.append( (k,W,H) )



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

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms
    #topic = H[topic_index]
    #return [all_terms[i] for i in topic.argsort()[-top:]]

k_values = []
coherences = []
terms = tfidf.get_feature_names()
for (k,W,H) in topic_models:
    # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    term_rankings = []
    for topic_index in range(k):
        #topic = H[topic_index]
        #terms = [tfidf.get_feature_names()[i] for i in topic.argsort()]
        term_rankings.append( get_descriptor( terms, H, topic_index, 10 ) )
    # Now calculate the coherence based on our Word2vec model
    k_values.append( k )
    coherences.append( calculate_coherence( w2v_model, term_rankings ) )
    print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )

df_NMF = pd.DataFrame({'Validation_Set':['NMF']*len(k_values), 'Topics':k_values, 'Coherence':coherences})
df_NMF.to_csv('nmf_tuning_results.csv', index=False)




