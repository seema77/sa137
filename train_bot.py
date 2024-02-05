import nltk 
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle 
import numpy as np 


words=[] #input from user "How are you?"
classes=[] #category
word_tags_list=[] #input ,category
ignore_words=['?','!',',','.',"'s","'m"] #

train_data_file=open('intents.json').read()
intents=json.loads(train_data_file)

def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words


for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern_word=nltk.word_tokenize(pattern) #how are you =>  ["how","are","you","?"]
        words.extend(pattern_word)
        word_tags_list.append((pattern_word,intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        stem_words=get_stem_words(words,ignore_words)
    
print(stem_words)
print("------------------------------->")
print(word_tags_list[10])
print("------------------------------->")
print(classes)
print("------------------------------->")
print(words)
