
# coding: utf-8

# In[3]:

import pandas as pd
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords
import nltk.data


# In[4]:

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


# In[5]:

#nltk.download('punkt')   
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence,               remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[6]:

train = pd.read_csv( "unlabeledTrainData1.tsv", header=0, 
 delimiter="\t", quoting=3 )


# In[7]:

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
y = 0
print len(train["review"])
for review in train["review"]:
    if(y % 1000 == 0):
        print "line:"
        print y
    y += 1
    sentences += review_to_sentences(review, tokenizer) 
    


# In[9]:

print len(sentences)


# In[11]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 160   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)


# In[13]:

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[12]:

# model.doesnt_match("man woman child kitchen".split())


# In[14]:

# model.doesnt_match("france england germany berlin".split())


# In[15]:

# model.doesnt_match("paris berlin london austria".split())


# In[16]:

# model.most_similar("man")


# In[17]:

# model.most_similar("queen")


# In[18]:

# model.most_similar("awful")


# In[19]:

# model['awful']


# In[20]:

print "Получены точки слов"


# In[33]:

model.wv.syn0

