"""
Created on Tue Feb 22 14:45:13 2022
@author: Valentinus R. Hananto
"""

import json
import pandas as pd
import numpy as np
import spacy
import tomotopy as tp
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from nltk.corpus import stopwords
from itertools import chain


def windowdiff(segment1, segment2):
    seg1 = ""
    element = 1
    for i in segment1:
        if(element==1):
            element = 0
        else:
            element = 1
        for i in range(i):
            seg1+=str(element)
    #print(seg1)
    
    seg2 = ""
    element = 1
    for i in segment2:
        if(element==1):
            element = 0
        else:
            element = 1
        for i in range(i):
            seg2+=str(element)
    #print(seg2)
    
    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    wd = 0
    k=round(len(seg1)/20)
    #print(k)
    for i in range(len(seg1) - k):
        wd += abs(len(set(seg1[i:i+k])) - len(set(seg2[i:i+k])))
    
    #print(wd)
    #print(len(seg1)-k)
    wd = wd/(len(seg1)-k)
    return wd

def pk(segment1, segment2):
    seg1 = ""
    element = 1
    for i in segment1:
        if(element==1):
            element = 0
        else:
            element = 1
        for i in range(i):
            seg1+=str(element)
    #print(seg1)
    
    seg2 = ""
    element = 1
    for i in segment2:
        if(element==1):
            element = 0
        else:
            element = 1
        for i in range(i):
            seg2+=str(element)
    #print(seg2)
    
    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    pk = 0
    k=round(len(seg1)/20)
    #k = 2
    #print("k="+str(k))
    for i in range(len(seg1) - k):
        a = seg1[i]==seg1[i+k]
        b = seg2[i]==seg2[i+k]
        #print(str(a)+" "+str(b))    
        pk += a^b
    
    #print(pk)
    #print(len(seg1)-k)
    pk = pk/(len(seg1)-k)
    return pk

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def preprocess(texts):
    # Tokenizing & Remove Stop Words
    word_list = [simple_preprocess(str(txt), min_len=3) for txt in texts]

    # Remove stopwords
    st_words = stopwords.words('english')
    word_list_nostops = [[word for word in txt if word not in st_words] for txt in word_list]

    # Create bigram models
    bigram = Phrases(word_list_nostops, min_count=10, threshold=100) 
    bigram_model = Phraser(bigram)

    # Form Bigrams
    data_words_bigrams = [bigram_model[w_vec] for w_vec in word_list_nostops]
    
    # Do lemmatization keeping only noun, adj, verb
    texts = []  
    for sent in data_words_bigrams:
        doc = nlp(" ".join(sent)) 
        texts.append([token.lemma_ for token in doc if token.lemma_ != '-PRON-' and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']])
    return texts

def write_segment(targets):
    #path = df.path.values[nmr]
    target_dict = {"segment" : targets}
    target_json = json.dumps(target_dict)
    with open('segment.json', 'a') as f:
        f.write(target_json)
        f.write("\n")

#returns segmented text
def TS(sentence,t,targets,DS):
    # for each sentences in a document
    n = len(sentence)
    for i in range(n):
        dist = 0
        segA = []
        segB = []
        b = i + h
        #print(n)
        if(b <= n - h):
            segA = sentence[0:b]
            segA = list(chain.from_iterable(sentence[0:b]))
            segB = sentence[b+1:b+h]
            segB = list(chain.from_iterable(sentence[b:b+h]))
            doc_inst = mdl.make_doc(segA)
            topic_dist1, log = mdl.infer(doc_inst)
            topic_idx1 = np.array(topic_dist1).argmax()
            #print(topic_dist1)
            doc_inst = mdl.make_doc(segB)
            topic_dist2, log = mdl.infer(doc_inst)
            topic_idx2 = np.array(topic_dist2).argmax()
            #print(topic_dist2)
            if(topic_idx1!=topic_idx2):
                #calculate Manhattan distance
                for k in range(mdl.k):
                    dist += abs(topic_dist1[k] - topic_dist2[k])
            if(dist>t):
                #print("new segment")
                targets.append(b)
                DS.append(segA)
                #print(sentence[b:n])
                #print("")
                DS, targets = TS(sentence[b:n],t,targets,DS)
                break
        else:
            targets.append(n)
            DS.append(list(chain.from_iterable(sentence[0:n])))
            break
    return(DS,targets)
         
def Search(t_seed,max_iter):
    d = 0.05
    t_opt = t_seed
    perpl_min = Segmentation(0,t_seed)
    for i in range(1,max_iter):
        t = t_seed + d
        perpl = Segmentation(i,t)
        if(perpl < perpl_min):
            t_opt = t
            perpl_min = perpl
        d = d + 0.05
    print("t opt: "+str(t_opt))

def Segmentation(i,t):
    DS = [] #to store segmented documents
    print("Iteration: "+str(i)+", t: "+str(t))
    with open('segment.json', 'w') as f:
        f.write("\n")
    doc_split = []
    for nmr in range(len(df)):
        #print(nmr)
        targets = [] #segment_length
        sentence = preprocess(filter(None,df.content.values[nmr].split('\n')))
        DS, targets = TS(sentence,t,targets,DS)
        write_segment(targets)
                    
    # retrain LDA model
    mdl2 = tp.LDAModel(k=140, rm_top=10, seed=123)
    # Add docs to train
    for vec in DS:
        mdl2.add_doc(vec)
    mdl2.train(1000)
    perpl = mdl2.perplexity
    print ("Perplexity: " + str(perpl))
    # Calculate segmentation errors
    segment1 = df.segment.values.tolist()
    df2 = pd.read_json('segment.json', lines = True)
    segment2 = df2.segment.values.tolist()
    total = 0
    for i in range(len(df2)):
        p=(pk(segment1[i],segment2[i]))
        total=total+p
    print("PK: " + str(total/len(df2)))
    total = 0
    for i in range(len(df2)):
        wd=(windowdiff(segment1[i],segment2[i]))
        total=total+wd
    print("WD: " + str(total/len(df2)))
    return(perpl)   




# read document
df = pd.read_json('../Data/choi3-5.json', lines = True)
print(df.shape)

# read LDA model
mdl = tp.LDAModel.load('../Data/Choi3-5.lda')

# TopicDiff-LDA initialization
h = 3
window_size = 3
w = window_size-1
t_seed = 1.7
max_iter = 5
Search(t_seed, max_iter)