### FIRST SECTION ###

import pandas as pd 
import numpy as np 
import re
import nltk
import string
import spacy 
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#import the dataframe and read it with candidates as index
df = pd.read_csv(r'C:\Users\Luca Nannini\Desktop/AllDebates.csv')
df.set_index("CANDIDATE", inplace= True)
df.head(10)
### FIRST DEBATE ###

first_debate = df.iloc[:131]
first_debate
#extract the TEXT columns as list
first_text = list(first_debate.TEXT)
print(first_text)
#using list comprehensions, lowercase the list
first_text = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in first_debate['TEXT'])
#convert text from list to string
first_text = ''.join(first_text)
#tokenize
first_tokens = nltk.word_tokenize(first_text)
#stopwords removal
stop = stopwords.words('english')
first_tokens = [token for token in first_tokens if token not in stop]
# remove words less than two letters
first_tokens = [word for word in first_tokens if len(word) >= 2]
preprocessed_first= ' '.join(first_tokens)
preprocessed_first
#let's now visualize the 10 most frequent words without stopwords
freq_dist_first = nltk.FreqDist(first_tokens)
freq_dist_first.most_common(10)
#POS_tag in NLTK
tagged_sent_first = nltk.pos_tag(first_tokens)
print(tagged_sent_first)
#Regular expression outputting only nouns as list 
all_nouns_first= []
for word, pos in tagged_sent_first:
    if pos in ['NN', 'NNP', 'NNS', 'NNPS'] :
        all_nouns.append(word)
print(all_nouns_first)
#lemmatize the nouns 
lmtzr = WordNetLemmatizer()
tokens_first = [lmtzr.lemmatize(word) for word in all_nouns_first]
#nouns frequencies
freq_nouns_first = nltk.FreqDist(all_nouns_first)
top_common_nouns_first = freq_nouns_first.most_common(20)
top_common_nouns_first


### CANDIDATE SPEECHES ###


#loc function to select rows of messages for TRUMP 
trump = first_debate.loc["TRUMP", "TEXT"]
#print the panda.core.series.series to list
trump = trump.tolist()
#convert to string, lowercase items, tokenize, remove stopwords
trump = ''.join(trump)
trump = (re.sub("[^A-Za-z']+", ' ', str(trump)).lower())
trump_tokens = nltk.word_tokenize(trump)
stop = stopwords.words('english')
trump_tokens = [token for token in trump_tokens if token not in stop]
trump_tokens
#remove words with less than 2 characters 
trump_tokens = [word for word in trump_tokens if len(word) >= 2]
# output a string for the cleaned tokens
preprocessed_text= ' '.join(trump_tokens)
#print out the 10 top words for frequency distribution
freq_dist = nltk.FreqDist(clean_words_trump)
freq_dist.most_common(10)

#pos_tag for evidence nouns
tagged_sent_trump = nltk.pos_tag(clean_words_trump)
print(tagged_sent_trump)
#detect all nouns (s/p & proper s/p nouns)
all_nouns_trump= []
for word, pos in tagged_sent_trump:
    if pos in ['NN', 'NNP', 'NNS', 'NNPS'] :
        all_nouns_trump.append(word)
print(all_nouns_trump)
#compute trump's top 20 frequent words
freq_trump_nouns = nltk.FreqDist(all_nouns_trump)
top_common_nouns_trump = freq_trump_nouns.most_common(20)
top_common_nouns_trump

# same process for Clinton

# loc function to select rows of messages
clinton = first_debate.loc["CLINTON", "TEXT"]
print(clinton)
#print the panda.core.series.series to list
list(clinton)
#convert to lowercase items in list and then convert to string
clinton = (re.sub("[^A-Za-z']+", ' ', str(clinton)).lower())
clinton = ''.join(clinton)
clinton
#tokenize the string words and output it as list
clean_words_clinton = word_tokenize(clinton)
clean_words_clinton
#remove stopwords
clean_words_clinton = [word for word in clean_words_clinton if word not in stopwords.words('english')]
print(clean_words_clinton)
#print out the 10 top words for frequency distribution
freq_dist = nltk.FreqDist(clean_words_clinton)
freq_dist.most_common(10)
#pos_tag for evidence nouns
tagged_sent_clinton = nltk.pos_tag(clean_words_clinton)
print(tagged_sent_clinton)
#detect all nouns (s/p & proper s/p nouns)
all_nouns_clinton= []
for word, pos in tagged_sent_clinton:
    if pos in ['NN', 'NNP', 'NNS', 'NNPS'] :
        all_nouns_clinton.append(word)
print(all_nouns_clinton)
#compute trump's top 20 frequent words
freq_clinton_nouns = nltk.FreqDist(all_nouns_clinton)
top_common_nouns_clinton = freq_clinton_nouns.most_common(20)
top_common_nouns_clinton

### TWEETS ### 

#import and visualize the dataframe
tweets = pd.read_csv(r'C:\Users\Luca Nannini\Desktop/TweetsClean.csv')
tweets.set_index("created_at", inplace= True)
tweets.head(20)
#reiterate over columns for visualize them 
for col in tweets: 
    print(col)
#filtering only original tweets, removing of Retweets messages
indexNames = tweets[tweets['RT'] == False]
indexNames
#creating a list object of just original tweets
tweet_list = list(indexNames.text)
tweet_list
#convert to string for starting the cleaning process 
tweet = ''.join(tweet_list)
#removal of hashtags/account
tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
#removal of punctuation
tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\_\=\*\@\#\$\"\''\``]", " ", tweet).split())
#remove hyperlinks
tweet = ' '.join(re.sub(r'https?:\/\/.*[\r\n]*', '', tweet).split())
#remove stock market tickers like $GE
tweet = ' '.join(re.sub("r'\$\W+\w*", " ", tweet).split())
#replace consecutive non-ASCII characters with a space
tweet = ' '.join(re.sub(r"[^\x00-\x7F]+", " ", tweet).split())
#removal of numbers
tweet = ' '.join(re.sub(r"(\d+)", " ", tweet).split())
#removal chains of the same character
tweet = ' '.join(re.sub(r'(.)\1{2,}', r'\1', tweet).split())
tweet = tweet.lower()  
#create a dictionary of slang words, abbreviations and contractions
def slang():
    
    return {"'s":"is",
        "s":"is",
        "'re":"are",
        "r":" are", 
        "'ll":"will",
        "ll":"will", 
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "i'm":"I am",
        "im":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "rn":"right now",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wtf":"what the fuck",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }
#create a string with standard words out of 'slang' dictionary and clean
SLANG = slang()
twtext = tweet.replace("’","'")
words = twtext.split()
reformed = [SLANG[word] if word in SLANG else word for word in words]
TW = " ".join(reformed)
TW = ''.join(TW)
#tokenize the words 
twtokens = nltk.word_tokenize(TW)
#remove stopwords
stop = stopwords.words('english')
twtokens = [token for token in twtokens if token not in stop]
#remove words less than two letters
twtokens = [word for word in twtokens if len(word) >= 3]
#lemmatize
lmtzr = WordNetLemmatizer()
twtokens = [lmtzr.lemmatize(word, pos="v") for word in twtokens]
#create a string of cleaned tweets tokens and save it
preprocessed_tweets = ' '.join(twtokens)
preprocessed_tweets
with open('tweetstextcleaned.txt','w') as f:
    f.write(preprocessed_tweets)

#print out the 10 top words for frequency distribution
freq = nltk.FreqDist(twtokens)
freq_words_tw = freq.most_common(50)
freq_words_tw
#pos_tag
tagged_sent_tweets = nltk.pos_tag(twtokens)
print(tagged_sent_tweets)
#filter only nouns
all_nouns_tweets= []
for word, pos in tagged_sent_tweets:
    if pos in ['NN', 'NNP', 'NNS', 'NNPS'] :
        all_nouns_tweets.append(word)
#top_common_nouns_tweets
freq_tweets_nouns = nltk.FreqDist(all_nouns_tweets)
top_common_nouns_tweets = freq_tweets_nouns.most_common(20)
top_common_nouns_tweets
#filter only verbs
all_verbs_tweets= []
for word, pos in tagged_sent_tweets:
    if pos in ['VB', 'VBG', 'VBN', 'VBP','VBZ'] :
        all_verbs_tweets.append(word)
print(all_verbs_tweets)
#top_common_verbs_tweets 
freq_tweets_verbs = nltk.FreqDist(all_verbs_tweets)
top_common_verbs_tweets = freq_tweets_verbs.most_common(20)
top_common_verbs_tweets


---------------------------------------------------------------------------------------
### SECOND SECTION ####

### COS SIM & SOFT COS SIM BETWEEN FIRST DEBATE AND TWEETS ###

import gensim
from gensim.matutils import softcossim 
from gensim import corpora
from gensim.utils import simple_preprocess
import os
from gensim.models import KeyedVectors
from gensim.downloader import base_dir
print(gensim.__version__)

import re
import sklearn
from pprint import pprint

#DEBATE sentences list - no string 

df = pd.read_csv(r'C:\Users\Luca Nannini\Desktop/AllDebates.csv')
df.set_index("CANDIDATE", inplace= True)
first_debate = df.iloc[:131]
debate = list(first_debate.TEXT)
# remove common words and tokenize
stoplist = stopwords.words('english')
debtok = [
     [word for word in document.lower().split() if word not in stoplist]
    for document in debate
 ]
# remove words that appear only once
frequency = defaultdict(int)
for text in debtok:
     for token in text:
        frequency[token] += 1       
debtok = [
     [token for token in text if frequency[token] > 1]
     for text in debtok
 ]
#remove punctuation marks
punctuation_marks = re.compile(r'[,.!|\/:?]+')

new_word_list = []
for words in debtok:
    sub_list = []
    for word in words:
        w = punctuation_marks.sub('', word)
        sub_list.append(w)
    new_word_list.append(sub_list)
#remove numbers
debate = [list(filter(None, [re.sub(r'\d+','', x) for x in y])) for y in
       new_word_list]

#TWEETS sentences list
tweets = pd.read_csv(r'C:\Users\Luca Nannini\Desktop/TweetsClean.csv')
tweets.set_index("created_at", inplace= True)
indexNames = tweets[tweets['RT'] == False]
tweets = list(indexNames.text)
#remove common words and tokenize
stoplist = stopwords.words('english')
twtok = [
     [word for word in document.lower().split() if word not in stoplist]
      for document in tweets
 ]
punctuation_marks = re.compile(r'[,.!|\/:?]+')
#remove punctuation marks
new_tw_list = []
for words in twtok:
    sub_tw = []
    for word in words:
        w = punctuation_marks.sub('', word)
        sub_tw.append(w)
    new_tw_list.append(sub_tw)
#remove numbers
tweets_cleaned = [list(filter(None, [re.sub(r'\d+','', x) for x in y])) for y in
       new_tw_list]
#remove user mentions
tweets_cleaned = [list(filter(None, [re.sub(r'(@[A-Za-z0-9]+)','', x) for x in y])) for y in
       tweets_cleaned]
#remove non-ASCII characters
tweets_cleaned = [list(filter(None, [re.sub(r'[^\x00-\x7F]+','', x) for x in y])) for y in
       tweets_cleaned]


####### LEXICON COMPARISON ########

### COS SIM: Counter & Tf-Idf ###

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
documents = [preprocessed_first, preprocessed_tweets] 

# 1.
# CounterVectorizer: Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  )
df
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(df, df))

# 2.
# TF-IDF: Create the Document Term Matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer()
sparse_matrix = tfidf_vectorizer.fit_transform(documents)
doc_term_matrix = sparse_matrix.todense()
tfidf = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf_vectorizer.get_feature_names(), 
                  )
tfidf
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfidf, tfidf))

# 3.
# Reiterate the two classes (Count & TF-IDF) with different thresholds of word frequencies for tweets tokens
from itertools import dropwhile
# [n] = set threshold (0,1,10,50,100)
for key, count in dropwhile(lambda key_count: key_count[1] > [n], counterB.most_common()):
    del counterB[key]
counterB = ' '.join(counterB)
# always check the different length 
len(counterB)

######## SEMANTIC COMPARISON #########

### Softcossim with FastText pre-trained model ###
 
import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
print(gensim.__version__)
# Download the FastText model
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
# Prepare a dictionary and a corpus.
documents = [tweets[n].split(), preprocessed_first.split()]
dictionary = corpora.Dictionary(documents)
# Convert the sentences into bag-of-words vectors.
debate_vec = dictionary.doc2bow(simple_preprocess(preprocessed_first))
tweets_vec = dictionary.doc2bow(simple_preprocess(tweets[n]))
# Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
# Compute soft cosine similarity
print(softcossim(debate_vec, tweets_vec, similarity_matrix))

# Reiterate the softcossim with different threshold of word frequencies for tweets tokens

----------------------------------------------------------------------------------------
###THIRD SECTION###

import pandas as pd
import numpy as np
import re
import pprint
from collection import defaultdict

import string
from string import punctuation

import os

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import gensim
from gensim import corpora, models, similarities

import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### TWEETS SEGMENT ANALYSIS ###

#no-retweets only
tweets = pd.read_csv(r'C:\Users\Luca Nannini\Desktop/TweetsClean.csv')
tweet = tweets[tweets['RT'] == False]
#import datetime [must be imported in the following cell]
from datetime import datetime
from datetime import timedelta
#create datetime object out of 'created_at' column values
clean_timestamp =  tweet['created_at'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ'))
#offset in hours for EST timezone 
offset_hours = -16
#account for offset from UTC using timedelta                                
local_timestamp = clean_timestamp + timedelta(hours=offset_hours)  
#print out the final timestamp visualizing only hour/minute/sec
final_timestamp = local_timestamp.dt.strftime('%X')
#create a new dataframe with the new time format and tweets text
TIME = final_timestamp
TEXT = tweet['text']
new_tweet = pd.DataFrame(dict(TIME = TIME, TEXT = TEXT))
#set as index Time for splitting according to the desired time frame
new_tweet.set_index("TIME", inplace= True)
new_tweet 

# N.b.: +5 seconds have been taken into account from the end of the last sentence produced by the speaking candidate at the end of the given topic segments

# Reiterate the histogram substituting:
# - 'tweets_ratio' with 'tweets_ratio_[n]'
# - 'new_tweet['TIME'] with 'TW_[n]['TIME']
# - layout title with the right topic segment name

#define "Achieving Prosperity" segment
TW_I = new_tweet.loc['09:04:53':'09:31:23']
TW_I.reset_index(inplace= True)
TW_I_TEXT = list(TW_I.TEXT)

#define "Candidate Figure Issues" segment
TW_II = new_tweet.loc['09:31:38':'09:43:41']
TW_II.reset_index(inplace= True)
TW_II_TEXT = list(TW_II.TEXT)

#define "America's direction" segment
TW_III = new_tweet.loc['09:44:06':'10:06:14']
TW_III.reset_index(inplace= True)
TW_III_TEXT = list(TW_III.TEXT)

#define "Securing America" segment
TW_IV = new_tweet.loc['10:06:26':'10:33:00']
TW_IV.reset_index(inplace= True)
TW_IV_TEXT = list(TW_IV.TEXT)

#define "Mutual and Election Acceptance"
TW_V = new_tweet.loc['10:33:04':'10:38:56']
TW_V_TEXT = list(TW_V.TEXT)
TW_V.reset_index(inplace= True)

### Visualize tweet volume during the overall debate time frame ###

# Reset index in order to parse the time column 
new_tweet.reset_index(inplace= True)
# Produce the histogram 
tweets_ratio = pd.to_datetime(new_tweet['TIME'], format='%X')
trace = go.Histogram(
    x=tweets_ratio,
    marker=dict(
        color='lightblue'
    ),
    opacity=0.75
)
layout = go.Layout(
    title='Overall Debate',
    height=450,
    width=1200,
    xaxis=dict(
        title='Time Segment for each bar = 29sec'
    ),
    yaxis=dict(
        title='Tweet Volume'
    ),
    bargap=0.2,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
#'Time Segment for each bar = 29sec'

### Preprocess tweet segments as list of token strings###
tweets[n] = list(TW_[n].TEXT)
twtok[n] = [
     [word for word in document.lower().split()]
      for document in tweets[n]
 ]
# remove words that appear only once
frequency = defaultdict(int)
for text in twtok[n]:
     for token in text:
        frequency[token] += 1
debtok[n] = [
     [token for token in text if frequency[token] > 1]
     for text in twtok[n]
 ]
# RE removing words with length <2 characters
tweets = [list(filter(None, [re.sub(r'\b\w{1,2}\b','', x) for x in y])) for y in
       debtok]
# RE removing numbers
tweets = [list(filter(None, [re.sub(r'\d+','', x) for x in y])) for y in
       tweets]
# RE removing users mentions 
tweets = [list(filter(None, [re.sub(r'(@[A-Za-z0-9]+)','', x) for x in y])) for y in
       tweets]
# RE removing punctuation
tweets = [list(filter(None, [re.sub(r'[\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\$\"]+','', x) for x in y])) for y in
       tweets]
# RE removing non-ASCII characters
tweets = [list(filter(None, [re.sub(r'[^\x00-\x7F]+','', x) for x in y])) for y in
       tweets]
# RE removing too long character chains (e.g. "gooooo"="go")
tweets = [list(filter(None, [re.sub(r'(.)\1{2,}',r'\1', x) for x in y])) for y in
       tweets]
#lemmatize and remove stopwords
lmtzr = WordNetLemmatizer()
cleaned_tweets = [
     [lmtzr.lemmatize(word) for word in document if word not in stoplist]
    for document in tweets
 ]
cleaned_tweets

### LDA on tweet segment ###
dictionary1 = corpora.Dictionary(cleaned_tweets)
corpus1 = [dictionary.doc2bow(text) for text in cleaned_tweets]

total_topics = 5
lda = models.LdaModel(corpus1, id2word=dictionary1, num_topics=total_topics)

lda.show_topics(total_topics,10)

### SPLIT AND REITERATE OVER THE FIVE TOPIC SEGMENTS ###



### SPLIT THE DEBATE IN THE FIVE TOPIC SEGMENT

#'Achieving Prosperity'
Deb_I = list(first_debate.loc['09:04:52':'09:30:11'].TEXT)

#'Candidate Figure Issues'
Deb_II = list(first_debate.loc['09:31:38':'09:41:42'].TEXT)

#'America's Direction'
Deb_III = list(first_debate.loc['09:44:06':'10:04:10'].TEXT)

#'Securing America'
Deb_IV = list(first_debate.loc['10:06:26':'10:31:51'].TEXT)

#'Mutual & Election Acceptance'
Deb_V = list(first_debate.loc['10:33:04':'10:37:43'].TEXT)

##### LATENT DIRICHLET ALLOCATION #####

# 1) Overall First Debate LDA - No BOWs

# 1.1) Preprocess: lowercase, tokenize, filter out numbers\punctuation\words less than two letters
debtok = [
     [word for word in document.lower().split()]
    for document in debate
 ]

debate = [list(filter(None, [re.sub(r'\d+','', x) for x in y])) for y in
       debtok]
# remove words less than two letters
debate = [list(filter(None, [re.sub(r'\W*\b\w{1,2}\b','', x) for x in y])) for y in
       debate]
# remove punctuation
debate = [list(filter(None, [re.sub("[\.\,\…\!\?\:\;\-\—\_\=\*\@\#\$\"\''\``]",'', x) for x in y]))
                                    for y in debate
# remove common words and tokenize
stoplist = stopwords.words('english')
stoplist = ['know','believe','you','really','lot','that','get','got','much','many','put','kind','thanks','thank','think','well','take','taken','going','go','things','maybe','something','yes','way','would','could','actually','almost','see','sean','called','thing','let','done','went','say','whether','said','look','one','like','also','good','new','ever','little','cannot','everything','lester','even','hannity'] + list(stoplist)
debate = [
     [word for word in document if word not in stoplist]
    for document in debate
 ]

# 1.2) Create a Dictionary and a corpus  
dictionary = corpora.Dictionary(debate)

corpus = [dictionary.doc2bow(text) for text in debate]
# 1.3) Build LDA model
total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

# 1.4) visualize
lda.show_topics(total_topics,10)


### SOME JUICY GRAPHS ###

### Tweet Volume Histogram, reiterate over single segments

tweets_ratio = pd.to_datetime(*tweet segment df*['TIME'], format='%X')

trace = go.Histogram(
    x=tweets_ratio,
    marker=dict(
        color='lightblue'
    ),
    opacity=0.75
)

layout = go.Layout(
    title='Overall Debate',
    height=450,
    width=1200,
    xaxis=dict(
        title='Time Segment for each bar = 29sec'
    ),
    yaxis=dict(
        title='Tweet Volume'
    ),
    bargap=0.2,
)

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)

### Visualize a clustertmap of cosine metric 

from collections import OrderedDict

data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}
#data_lda

df_lda = pd.DataFrame(data_lda)
print(df_lda.shape)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
g=sns.clustermap(df_lda.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(10, 10))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

### pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='TSNE')
panel

### Wordcloud with topic histograms
#1.)

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(background_color='white',
                  width=2000,
                  height=1400,
                  max_words=15,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda.show_topics(formatted=False)

fig, axes = plt.subplots(5, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=200)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#2.)

from collections import Counter
topics = lda.show_topics(formatted=False)
data_flat = [w for w_list in cleaned_tweets for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(5, figsize=(20,30), sharey=True, dpi=160)
cols = [color for name, color in mcolors.XKCD_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.9, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.4, label='Volume')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.090); ax.set_ylim(0, 500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
# Do not forget to attune the parameters :)
fig.tight_layout(w_pad=2)    
fig.suptitle('Tweets overall LDA topics', fontsize=18, y=1.05)    
plt.show()

### FastText Debate Word Embeddings with t-SNE

from gensim.models import FastText
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

model = FastText(cleaned_tweets, size=100, window=50, min_count=500, workers=6)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)

def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
tsne_plot(model)

### t-SNE cluster 

# Get topic weights and dominant topics 
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda[corpus]):
    topic_weights.append([w for i, w in row_list[0]])
# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values
# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)
# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, n_iter=2500, perplexity=40, random_state=23, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)
# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of 5 LDA Tweets Topics", 
              plot_width=1200, plot_height=600)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)