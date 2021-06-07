import csv
from random import randrange
from random import randint
import random
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import os
import json


text = 'text'
issue = 'issue'
label = 'label'
w2v_file = '/Users/zms/Documents/Study/cs577/w2v.bin'
data_folder_name = 'data'

def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2label

def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2label[tweet_id]])

def handle_text(input_dict):

    for key in input_dict:
        cur_text = input_dict[key]

        #print(cur_text)

        lower_alphabetic_text = re.sub('[^A-Za-z]', ' ', cur_text).lower()
        tokenized_text = word_tokenize(lower_alphabetic_text)
        for word in tokenized_text:
            if word in stopwords.words('english'):
                tokenized_text.remove(word)

        for i in range(len(tokenized_text)):
            tokenized_text[i] = PorterStemmer().stem(tokenized_text[i])

        #print(tokenized_text)

        input_dict[key] = tokenized_text

    #print(input_dict)
    return input_dict

def wrap_data_dict(preprocessed_text, train_tweet_id2issue, train_tweet_id2label):

    output_dict = {}

    for key in preprocessed_text:
        cur_dict = {}
        cur_dict[text] = preprocessed_text[key]
        cur_dict[issue] = train_tweet_id2issue[key]
        cur_dict[label] = train_tweet_id2label[key]

        output_dict[key] = cur_dict

    return output_dict

def create_word_dict(input_vector_dict,data_type):

    word_list = []
    for key in input_vector_dict:
        cur_vector = input_vector_dict[key][data_type]

        for each_word in cur_vector:
            if each_word not in word_list:
                word_list.append(each_word)

    #add one element to handle unknown word
    
    word_list.append('unknown')
    word_to_ix = {word: i for i, word in enumerate(word_list)}
    return word_to_ix

#==============================================================#
#======================Below for BoW===========================#
#==============================================================#

def one_hot_encoding_for_one_vector(input_vector,word_dict):
    output_vector = np.zeros(len(word_dict))

    for item in input_vector:
        if item in word_dict:
            output_vector[word_dict[item]] += 1
        else:
            output_vector[word_dict['unknown']] += 1
    return output_vector

def one_hot_encoding_for_dict(input_vector_dict,data_type,word_dict):
    for key in input_vector_dict:
        old_vector = input_vector_dict[key][data_type]
        new_vector =one_hot_encoding_for_one_vector(old_vector,word_dict)
        input_vector_dict[key][data_type] = new_vector

    #print(input_vector_dict)
    return input_vector_dict

def get_preprocessed_x_matrix(wrapped_dict):

    x_matrix = []
    y_matrix = []

    key_list = []
    label_type_list = []

    for key in wrapped_dict:
        corresponding_text = wrapped_dict[key][text]
        corresponding_issue = wrapped_dict[key][issue]
        if(wrapped_dict[key][label] == None):
            corresponding_label = None
        else:
            corresponding_label = int(wrapped_dict[key][label])-1

        corresponding_x_i = np.concatenate((corresponding_text, corresponding_issue))

        x_matrix.append(corresponding_x_i)
        y_matrix.append(corresponding_label)
        key_list.append(key)
        if corresponding_label not in label_type_list:
            label_type_list.append(corresponding_label)

    '''
    print(x_matrix)
    print(y_matrix)
    print(np.array(x_matrix).shape)
    print(np.array(y_matrix).shape)
    '''
    
    return x_matrix,y_matrix,key_list,label_type_list

def get_all_dict_to_ix(wrapped_dict):
    text_word_dict_to_ix = create_word_dict(wrapped_dict,text)
    issue_word_dict_to_ix = create_word_dict(wrapped_dict,issue)

    return text_word_dict_to_ix,issue_word_dict_to_ix

def BoW(wrapped_dict,text_word_dict_to_ix,issue_word_dict_to_ix):

    wrapped_dict = one_hot_encoding_for_dict(wrapped_dict,text,text_word_dict_to_ix)

    wrapped_dict = one_hot_encoding_for_dict(wrapped_dict,issue,issue_word_dict_to_ix)

    '''
    print(wrapped_dict)
    print(len(wrapped_dict[0][text]))
    print(len(wrapped_dict[0][issue]))
    print(len(wrapped_dict[0][author_label]))
    print(text_word_list)
    print(issue_word_list)
    print(author_word_list)
    exit()
    '''
    x_matrix,y_matrix,key_list,label_type_list = get_preprocessed_x_matrix(wrapped_dict)


    return x_matrix, y_matrix, key_list, label_type_list

#==============================================================#
#====================Below for Embedding=======================#
#==============================================================#

def embeddings_lookup_table(input_dict,vocab_size,embedding_size,word_embedding):

    #word_embedding = nn.Embedding(vocab_size, embedding_size)

    ix_to_embedding = {}

    for key in input_dict:
        ix_to_embedding[input_dict[key]] = word_embedding(torch.LongTensor([input_dict[key]]))

    return ix_to_embedding

def word_to_embedding(word_to_ix,ix_to_embedding):

    word_to_embedding = {}
    for key in word_to_ix:
        word_to_embedding[key] = ix_to_embedding[word_to_ix[key]]

    return word_to_embedding

def wrap_dict_to_separate_list(wrapped_dict):

    text_list = []
    issue_list = []
    key_list = []
    label_list = []
    label_type_list = []

    for key in wrapped_dict:
        text_list.append(wrapped_dict[key][text])
        issue_list.append(wrapped_dict[key][issue])
        key_list.append(key)
        if(wrapped_dict[key][label] == None):
            label_list.append(None)
        else:
            label_list.append(int(wrapped_dict[key][label])-1)
    
    
        cur_label = wrapped_dict[key][label]
        if cur_label not in label_type_list:
            label_type_list.append(cur_label)

    return text_list,issue_list,key_list,label_list,label_type_list

def wrap_embedding_training_set(input_list,embeddings_dict):

    output_list = []

    for i in range(len(input_list)):
        cur_word_list = input_list[i]
        temp = []
        for each_word in cur_word_list:
            if(each_word in embeddings_dict):
                temp.append(embeddings_dict[each_word])
            else:
                temp.append(embeddings_dict['unknown'])
        output_list.append(temp)

    return output_list

def build_vocab():
	# Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2label = ReadFile('train.csv')
    wrapped_small_dict = wrap_data_dict(handle_text(train_tweet_id2text), 
    handle_text(train_tweet_id2issue), 
    train_tweet_id2label)
    small_text_word_dict_to_ix, _ = get_all_dict_to_ix(wrapped_small_dict)
    print('Dict size of labeled data: ', len(small_text_word_dict_to_ix))
    # add unlabeled text
    unlabeled_text, unlabeled_issue, unlabeled_label = read_unlabeled_data()
    train_tweet_id2text.update(handle_text(unlabeled_text))
    train_tweet_id2issue.update(handle_text(unlabeled_issue))
    train_tweet_id2label.update(unlabeled_label)
    wrapped_all_dict = wrap_data_dict(train_tweet_id2text, train_tweet_id2issue, train_tweet_id2label)
    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_all_dict)

    # text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
    # issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)

    # text_ix_to_embedding = embeddings_lookup_table(text_word_dict_to_ix,len(text_word_dict_to_ix),embedding_size,text_embedding)
    # issue_ix_to_embedding = embeddings_lookup_table(issue_word_dict_to_ix,len(issue_word_dict_to_ix),embedding_size,issue_embedding)
        
    # text_to_embedding = word_to_embedding(text_word_dict_to_ix,text_ix_to_embedding) # word to embedding
    # issue_to_embedding = word_to_embedding(issue_word_dict_to_ix,issue_ix_to_embedding)
    print('Dict size of all data: ', len(text_word_dict_to_ix))
    w2v = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    print('Pre-trianed Word2vec size:', len(w2v.index_to_key), ' embedding dim:', w2v.vector_size)

    oov = 0
    word_to_embedding = {}
    for key in text_word_dict_to_ix:
        if key in w2v.key_to_index:
            word_to_embedding[key] = torch.tensor(w2v.get_vector(key))
        else:
            oov += 1
            if key in small_text_word_dict_to_ix:
                word_to_embedding[key] = torch.tensor(np.random.uniform(size=(w2v.vector_size)))

    print('OOV size:', oov)
    print('Final w2v size: ', len(word_to_embedding))

    torch.save(word_to_embedding, 'word_to_embedding.pkl')
    # text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
    # text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)  # text_embedding_list[i] is word embeddings of sentence i
    # issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)




def read_unlabeled_data():

    unlabeled_text = {}
    unlabeled_issue = {}
    unlabeled_label = {}
    
    temp_label_dict = {}
    with open('data.json', 'r') as fi:
        [tweet_ids, tweet_id2topic]=json.load(fi)
        for key in tweet_id2topic:
            temp_label_dict[int(key)] = tweet_id2topic[key]

    print("Reading the unlabeled data-sets...")
    data_folder = data_folder_name
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for cur_file in filenames:
            #print(cur_file)
            extra_data_file = open(data_folder + '/' + cur_file,)
            extra_data = json.load(extra_data_file)
            for each_data in extra_data:
                if 'id' in each_data:
                    cur_id = int(each_data['id'])
                    cur_text = each_data[text]
                    if(cur_id in temp_label_dict):
                        unlabeled_text[int(cur_id)] = cur_text
                        unlabeled_issue[int(cur_id)] = temp_label_dict[cur_id]
                        unlabeled_label[int(cur_id)] = None
      
    '''  
    print(unlabeled_text)
    print(unlabeled_issue)
    print(len(unlabeled_text),len(unlabeled_issue))
    exit()
    '''

    # Closing file
    extra_data_file.close()

    copied_text = unlabeled_text.copy()
    '''
    Implement your Neural Network classifier here
    '''
    return unlabeled_text, unlabeled_issue, unlabeled_label

if __name__ == '__main__':
	build_vocab()

