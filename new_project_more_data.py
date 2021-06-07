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

def handle_data_with_word_embedding(text_embedding_list,issue_embedding_list,
    label_list,embedding_way_for_diff_length_sentence,Max_padding_size):

    context = None
    #label = None
    label_index = None

    Max_length = 0
    Min_length = 0
    for i in range(len(text_embedding_list)):
        cur_length = len(text_embedding_list[i])
        if(i == 0):
            Max_length = cur_length
        else:
            if(cur_length>Max_length):
                Max_length = cur_length
            if(cur_length<Min_length):
                Min_length = cur_length
    
    Min_length = 10
    for i in range(len(text_embedding_list)):

        cur_context = None
        cur_text_embedding_list = text_embedding_list[i]
        cur_issue_embedding_list = issue_embedding_list[i]

        if(embedding_way_for_diff_length_sentence == 'Average'):
            count = 0
            for j in range(len(cur_text_embedding_list)):
                count+=1
                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                else:
                    cur_context = cur_context+cur_text_embedding_list[j]
            cur_context = cur_context/count
        elif(embedding_way_for_diff_length_sentence == 'Sum'):
            for j in range(len(cur_text_embedding_list)):
                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                else:
                    cur_context = cur_context+cur_text_embedding_list[j]
        elif(embedding_way_for_diff_length_sentence == 'Min'):
            Min_value = 0
            Min_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Min_value = cur_norm
                    Min_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm < Min_value):
                        Min_value = cur_norm
                        Min_tensor = cur_text_embedding_list[j]
            cur_context = Min_tensor
        elif(embedding_way_for_diff_length_sentence == 'Max'):
            Max_value = 0
            Max_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Max_value = cur_norm
                    Max_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm > Max_value):
                        Max_value = cur_norm
                        Max_tensor = cur_text_embedding_list[j]
            cur_context = Max_tensor
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            Min_value = 0
            Max_value = 0
            Min_tensor = None
            Max_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Max_value = cur_norm
                    Min_value = cur_norm
                    Min_tensor = cur_text_embedding_list[j]
                    Max_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm > Max_value):
                        Max_value = cur_norm
                        Max_tensor = cur_text_embedding_list[j]
                    if(cur_norm < Min_value):
                        Min_value = cur_norm
                        Min_tensor = cur_text_embedding_list[j]

            cur_context = torch.cat((Min_tensor,Max_tensor),1)
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):

            #for j in range(Max_length):
            for j in range(Max_padding_size):

                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                elif(j < len(cur_text_embedding_list)):
                    cur_context = torch.cat((cur_context,cur_text_embedding_list[j]),1)
                else:
                    cur_context = torch.cat((cur_context,torch.FloatTensor([np.zeros((embedding_size))])),1)
                    #cur_context = torch.cat((cur_context,torch.FloatTensor(np.zeros((embedding_size)))),1)

        elif(embedding_way_for_diff_length_sentence == 'CuttingByMin'):

            for j in range(Min_length):

                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                elif(j>=len(cur_text_embedding_list)):
                    cur_context = torch.cat((cur_context,torch.FloatTensor([np.zeros((embedding_size))])),1)
                else:
                    cur_context = torch.cat((cur_context,cur_text_embedding_list[j]),1)

        elif(embedding_way_for_diff_length_sentence == 'Sentence'):

            cur_context = cur_text_embedding_list
            
        else:
            print("Please set a feasible way to form context vector: Average, Min, Max, MinMax...")
        
        for j in range(len(cur_issue_embedding_list)):
            #count+=1
            #cur_context = cur_context+cur_issue_embedding_list[j]
            #print(cur_context)
            #print(cur_issue_embedding_list[j])
            #print(torch.cat((cur_context, cur_issue_embedding_list[j]),1))
            
            cur_context = torch.cat((cur_context, cur_issue_embedding_list[j]),1)

        if(label_list[i] == None):
            cur_target_index = torch.LongTensor([0])
        else:
            cur_target_index = torch.LongTensor([label_list[i]])
        #cur_target_index = torch.LongTensor(label_list[i])

        if(i == 0):

            context = cur_context
            label_index = cur_target_index

        else:
            context=torch.cat((context, cur_context),0)
            label_index=torch.cat((label_index,cur_target_index),0) 

        #print(cur_context)
        #print(cur_context/count)
        #print(context)

    return context,label_index

def get_embedding_form(wrapped_dict,embedding_size,embedding_way_for_diff_length_sentence,Max_padding_size,text_embedding,issue_embedding):
    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_dict)

    if use_pretrained:
        text_to_embedding = torch.load(vocab_file)

    else:
        #text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
        text_ix_to_embedding = embeddings_lookup_table(text_word_dict_to_ix,len(text_word_dict_to_ix),embedding_size,text_embedding)
        text_to_embedding = word_to_embedding(text_word_dict_to_ix,text_ix_to_embedding)
    
    #issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)
    issue_ix_to_embedding = embeddings_lookup_table(issue_word_dict_to_ix,len(issue_word_dict_to_ix),embedding_size,issue_embedding)
    issue_to_embedding = word_to_embedding(issue_word_dict_to_ix,issue_ix_to_embedding)

    text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
    text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)  # text_embedding_list[i] is word embeddings of sentence i
    issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)

    context,label_index = handle_data_with_word_embedding(text_embedding_list,issue_embedding_list,label_list,
        embedding_way_for_diff_length_sentence,Max_padding_size)

    #print(context[:3])
    #print(context[:3].shape)
    #print(context.shape)
    #exit()
    return context,label_index,key_list,label_list,label_type_list,text_to_embedding,issue_to_embedding

def handle_data_with_word_embedding_2d(text_embedding_list,issue_embedding_list,
    label_list,embedding_way_for_diff_length_sentence,Max_padding_size):

    context = None
    #label = None
    label_index = None

    Max_length = 0
    Min_length = 0
    for i in range(len(text_embedding_list)):
        cur_length = len(text_embedding_list[i])
        if(i == 0):
            Max_length = cur_length
        else:
            if(cur_length>Max_length):
                Max_length = cur_length
            if(cur_length<Min_length):
                Min_length = cur_length
    
    Min_length = 10
    for i in range(len(text_embedding_list)):

        cur_context = None
        cur_text_embedding_list = text_embedding_list[i]
        cur_issue_embedding_list = issue_embedding_list[i]

        if(embedding_way_for_diff_length_sentence == 'Average'):
            count = 0
            for j in range(len(cur_text_embedding_list)):
                count+=1
                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                else:
                    cur_context = cur_context+cur_text_embedding_list[j]
            cur_context = cur_context/count
        elif(embedding_way_for_diff_length_sentence == 'Sum'):
            for j in range(len(cur_text_embedding_list)):
                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                else:
                    cur_context = cur_context+cur_text_embedding_list[j]
        elif(embedding_way_for_diff_length_sentence == 'Min'):
            Min_value = 0
            Min_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Min_value = cur_norm
                    Min_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm < Min_value):
                        Min_value = cur_norm
                        Min_tensor = cur_text_embedding_list[j]
            cur_context = Min_tensor
        elif(embedding_way_for_diff_length_sentence == 'Max'):
            Max_value = 0
            Max_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Max_value = cur_norm
                    Max_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm > Max_value):
                        Max_value = cur_norm
                        Max_tensor = cur_text_embedding_list[j]
            cur_context = Max_tensor
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            Min_value = 0
            Max_value = 0
            Min_tensor = None
            Max_tensor = None
            for j in range(len(cur_text_embedding_list)):
                
                if(j == 0):
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    Max_value = cur_norm
                    Min_value = cur_norm
                    Min_tensor = cur_text_embedding_list[j]
                    Max_tensor = cur_text_embedding_list[j]
                else:
                    cur_norm = torch.norm(cur_text_embedding_list[j])
                    if(cur_norm > Max_value):
                        Max_value = cur_norm
                        Max_tensor = cur_text_embedding_list[j]
                    if(cur_norm < Min_value):
                        Min_value = cur_norm
                        Min_tensor = cur_text_embedding_list[j]

            cur_context = torch.cat((Min_tensor,Max_tensor),0)
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):

            #for j in range(Max_length):
            for j in range(Max_padding_size):

                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                elif(j < len(cur_text_embedding_list)):
                    cur_context = torch.cat((cur_context,cur_text_embedding_list[j]),0)
                else:
                    cur_context = torch.cat((cur_context,torch.FloatTensor([np.zeros((embedding_size))])),0)
                    #cur_context = torch.cat((cur_context,torch.FloatTensor(np.zeros((embedding_size)))),1)

        elif(embedding_way_for_diff_length_sentence == 'CuttingByMin'):

            for j in range(Min_length):

                if(j == 0):
                    cur_context = cur_text_embedding_list[j]
                elif(j>=len(cur_text_embedding_list)):
                    cur_context = torch.cat((cur_context,torch.FloatTensor([np.zeros((embedding_size))])),1)
                else:
                    cur_context = torch.cat((cur_context,cur_text_embedding_list[j]),0)

        elif(embedding_way_for_diff_length_sentence == 'Sentence'):

            cur_context = cur_text_embedding_list
            
        else:
            print("Please set a feasible way to form context vector: Average, Min, Max, MinMax...")
        
        for j in range(len(cur_issue_embedding_list)):
            #count+=1
            #cur_context = cur_context+cur_issue_embedding_list[j]
            cur_context = torch.cat((cur_context, cur_issue_embedding_list[j]),0)


        if(label_list[i] == None):
            cur_target_index = torch.LongTensor([0])
        else:
            cur_target_index = torch.LongTensor([label_list[i]])
        #cur_target_index = torch.LongTensor(label_list[i])

        #print(cur_context)
        #print(cur_context.shape)
        #exit()
        if(i == 0):
            context = torch.FloatTensor([np.zeros((cur_context.shape))])
            context[0] += cur_context

            label_index = cur_target_index

        else:
            temp = torch.FloatTensor([np.zeros((cur_context.shape))])
            temp[0] += cur_context
            context=torch.cat((context, temp),0)
            label_index=torch.cat((label_index,cur_target_index),0) 

    #print(context)
    #print('data shape',context.shape)
    #print('Label shape',label_index.shape)
    #exit()
    return context,label_index

def get_embedding_form_2d(wrapped_dict,embedding_size,embedding_way_for_diff_length_sentence,Max_padding_size,text_embedding,issue_embedding):
    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_dict)

    if use_pretrained:
        text_to_embedding = torch.load(vocab_file)
        for key in text_to_embedding:
            text_to_embedding[key] = text_to_embedding[key].view(1, -1)

    else:
        #text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
        text_ix_to_embedding = embeddings_lookup_table(text_word_dict_to_ix,len(text_word_dict_to_ix),embedding_size,text_embedding)
        text_to_embedding = word_to_embedding(text_word_dict_to_ix,text_ix_to_embedding)
    
    #issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), issue_embedding_size)
    issue_ix_to_embedding = embeddings_lookup_table(issue_word_dict_to_ix,len(issue_word_dict_to_ix),issue_embedding_size,issue_embedding)
    issue_to_embedding = word_to_embedding(issue_word_dict_to_ix,issue_ix_to_embedding)

    text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
    text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)
    issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)

    context,label_index = handle_data_with_word_embedding_2d(text_embedding_list,issue_embedding_list,label_list,
        embedding_way_for_diff_length_sentence,Max_padding_size)

    #print(context[:3])
    #print(context[:3].shape)
    #print(context.shape)
    #exit()
    return context,label_index,key_list,label_list,label_type_list,text_to_embedding,issue_to_embedding

#==============================================================#
#===================Below for Bow&Embedding====================#
#==============================================================#

def get_bow_embedding_form(wrapped_dict,embedding_size,embedding_way_for_diff_length_sentence):

    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_dict)

    text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
    text_ix_to_embedding = embeddings_lookup_table(text_word_dict_to_ix,len(text_word_dict_to_ix),embedding_size,text_embedding)   
    text_to_embedding = word_to_embedding(text_word_dict_to_ix,text_ix_to_embedding)
    text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
    text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)

    issue_encoding_list = []
    for i in range(len(issue_list)):
        issue_encoding_list.append([torch.FloatTensor([one_hot_encoding_for_one_vector(issue_list[i],issue_word_dict_to_ix)])])
    
    context,label_index = handle_data_with_word_embedding(text_embedding_list,issue_encoding_list,label_list,
        embedding_way_for_diff_length_sentence)

    #print(context[:3])
    #exit()
    return context,label_index,key_list,label_list,label_type_list,text_to_embedding

#==============================================================#
#======================Below for Doc2Vec=======================#
#==============================================================#

def get_sentence_vector_lookup_table(train_tweet_id2text,embedding_size):

    sentence_list = []

    for key in train_tweet_id2text:
        sentence_list.append(train_tweet_id2text[key])
    #print(sentence_list)

    if not os.path.exists('Doc2Vec.model'):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentence_list)]

        max_epochs = 100
        vec_size = embedding_size
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        epochs=50,
                        dm =1)
          
        model.build_vocab(tagged_data)
        '''
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        '''
        model.save("Doc2Vec.model")
    else:
        model = Doc2Vec.load("Doc2Vec.model")

    output_vector_dict = {}
    for key in train_tweet_id2text:
        cur_text = train_tweet_id2text[key]
        output_vector_dict[key] = torch.FloatTensor([model.infer_vector(word_tokenize(cur_text.lower()))])
        #output_vector_dict[key] = torch.FloatTensor(model.infer_vector(word_tokenize(cur_text.lower())))

    return output_vector_dict,model

def get_embedding_form_Doc2Vec(wrapped_dict,embedding_size,text_vector_dict,embedding_way_for_diff_length_sentence):

    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_dict)

    issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)
    issue_ix_to_embedding = embeddings_lookup_table(issue_word_dict_to_ix,len(issue_word_dict_to_ix),embedding_size,issue_embedding)
    issue_to_embedding = word_to_embedding(issue_word_dict_to_ix,issue_ix_to_embedding)
    


    text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
    
    text_vector_list = []
    for cur_key in key_list:
        text_vector_list.append(text_vector_dict[cur_key])

    issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)

    context,label_index = handle_data_with_word_embedding(text_vector_list,issue_embedding_list,label_list,
        embedding_way_for_diff_length_sentence)

    #print(context)
    #exit()
    return context,label_index,key_list,label_list,label_type_list,issue_to_embedding

#==============================================================#
#======================Below for MLP Model=====================#
#==============================================================#

def partition(x_matrix,y_matrix,key_list,train_ratio,validation_ratio,test_ratio):

    length = len(x_matrix)
    train_index = int(length*train_ratio)
    validation_index = int(length*(train_ratio+validation_ratio))

    train_x = x_matrix[:train_index]
    train_x = torch.FloatTensor(train_x)
    train_y = y_matrix[:train_index]
    train_y = torch.LongTensor(train_y)

    validation_x = x_matrix[train_index+1:validation_index]
    validation_x = torch.FloatTensor(validation_x)
    validation_y = y_matrix[train_index+1:validation_index]
    validation_y = torch.LongTensor(validation_y)

    test_x = x_matrix[validation_index+1:]
    test_x = torch.FloatTensor(test_x)
    test_y = y_matrix[validation_index+1:]
    test_y = torch.LongTensor(test_y)

    #print(train_x,train_y)
    return train_x,train_y,validation_x,validation_y,test_x,test_y

class NeuralNet(nn.Module):  

    def __init__(self, total_embedding_size, label_num,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size):
        super(NeuralNet, self).__init__()

        self.linear1 = nn.Linear(total_embedding_size, 100)
        self.linear2 = nn.Linear(100, label_num)
        self.act = nn.ReLU()

        self.text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
        self.issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)
        
        #self.linear3 = nn.Linear(25, label_num)

    def forward(self, x):

        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x

class NeuralNet_2(nn.Module):  

    def __init__(self, total_embedding_size, label_num,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size):
        super(NeuralNet_2, self).__init__()

        self.linear1 = nn.Linear(total_embedding_size, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, label_num)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        #self.linear3 = nn.Linear(25, label_num)
        self.text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
        self.issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)

    def forward(self, x):

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = F.log_softmax(x, dim=-1)
        return x

class CNN(nn.Module):
    def __init__(
        self, 
        input_dim,          # D
        input_length,       # T
        kernel_size_list,   # [k_1, k_2, ... ]
        output_dim,         # num of classes
        dropout,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size
    ):
        super(CNN,self).__init__()
        
        hid_dim = hidden_dim  # TODO: try 
        kernel_size_list = kernel_size_list
        num_filters = len(kernel_size_list)
        
        # self.emb2hid = nn.Linear(input_dim, hid_dim) # D to H

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=hid_dim, 
                kernel_size=(k, input_dim)) 
            for k in kernel_size_list
        ])
        
        self.hid2cls = nn.Linear(num_filters*hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.text_embedding = nn.Embedding(len(text_word_dict_to_ix), embedding_size)
        self.issue_embedding = nn.Embedding(len(issue_word_dict_to_ix), embedding_size)

    def forward(self, x):
        """x: [B, T, D] """
        B, T, D = x.size()

        x = x.unsqueeze(1)  # [B, 1, T, D]
        # step1: project to hidden dim
        # x = self.emb2hid(x) # [B, 1, T=101, H=32]
        
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # shapes: [861, 32, 99], [861, 32, 98], [861, 32, 97]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [B, num_filters]
        # shapes: 
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        x = self.hid2cls(cat)
        x = F.log_softmax(x, dim=-1)
        return x

def train(model,train_x,train_y,validation_x,validation_y,label_type_list,learning_rate,weight_decay_rate,number_of_epoches,
    embedding_size,whether_analyze,num_of_record_results,model_type,loss_function_choice,update_method,
    wrapped_train_dict,embedding_way_for_diff_length_sentence,Max_padding_size,text_embedding,issue_embedding):


    if(loss_function_choice == 'NLLLoss'):
        loss_function = nn.NLLLoss()
    elif(loss_function_choice == 'CrossEntropyLoss'):
        loss_function = nn.CrossEntropyLoss()
    else:
        print("Please select a loss function: NLL, CrossEntropyLoss, ...")
    

    if(update_method == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif(update_method == 'Adam'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        print("Please set a feasible update method: SGD, Adam, ...")


    divided_size = int(number_of_epoches/num_of_record_results)
    train_accu_array = []
    validation_accu_array = []
    epoch_num_array = []

    if(whether_analyze==1):
        print("#Epoch,  loss,     training accuracy,    validation accuray")
    for epoch in range(number_of_epoches):
    
        model.zero_grad()
        log_probs = model(train_x)
        loss = loss_function(log_probs, train_y)
        loss.backward(retain_graph=True)
        optimizer.step()

        #print(train_x[0])
        #print(model.text_embedding(torch.LongTensor([1])))

        if(data_format == 'WordEmbedding'):
            x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
                    Max_padding_size,text_embedding,issue_embedding)
            train_x,train_y,validation_x,validation_y,test_x,test_y = partition(x_matrix,y_matrix,key_list,train_ratio,validation_ratio,test_ratio)
    
        elif(data_format == 'NoFlattenEmbedding'):
            x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form_2d(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
                    Max_padding_size,text_embedding,issue_embedding)
            train_x,train_y,validation_x,validation_y,test_x,test_y = partition(x_matrix,y_matrix,key_list,train_ratio,validation_ratio,test_ratio)
    
        #print(key_list[0])
        if(whether_analyze==1):
            if(epoch%divided_size == 0 or epoch == number_of_epoches-1):
                cur_train_accuracy = evaluate(model,train_x,train_y)
                cur_validation_accuracy = evaluate(model,validation_x,validation_y)

                train_accu_array.append(cur_train_accuracy)
                validation_accu_array.append(cur_validation_accuracy)
                epoch_num_array.append(epoch)
                
                print(epoch, loss.data.item(),cur_train_accuracy,cur_validation_accuracy)

    return model,train_accu_array,validation_accu_array,epoch_num_array

def prediction(model,x):

    predict_y = []
    correct = 0
    for i in range(len(x)):
        input = torch.unsqueeze(x[i], 0)
        cur_predict_y = torch.argmax(torch.squeeze(model.forward(input)))

        predict_y.append(int(cur_predict_y.data.item()))


    return predict_y

def calculate_accuracy(predict_y,actual_y):

    correct = 0
    for i in range(len(predict_y)):

        if(predict_y[i] == actual_y[i]):
            correct += 1

    accuracy = correct/len(predict_y)

    #print(predict_y,y,correct/len(x))

    return accuracy

def evaluate(model,x,y):

    predict_y = prediction(model,x)
    actual_y = y.numpy()
    accuracy = calculate_accuracy(predict_y,actual_y)

    return accuracy

def plot_one_x_vs_two_y(x,y_1,y_2,x_label_name,y_label_name,y_1_name,y_2_name,file_name):

    plt.plot(x, y_1, label=y_1_name)
    plt.plot(x, y_2, label=y_2_name)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.legend()
    #plt.savefig('F1score_diff_lr_op'+str(args.option)+'_sgd.png')
    plt.savefig(file_name+'.png')
    plt.close()

    return

def first_train(data_format,model_type,train_ratio,validation_ratio,test_ratio,
    learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
    whether_analyze,num_of_record_results,image_folder,update_method,
    loss_function_choice,embedding_way_for_diff_length_sentence,Max_padding_size):

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2label = ReadFile('train.csv')

    copied_train_text = train_tweet_id2text.copy()
    '''
    Implement your Neural Network classifier here
    '''

    wrapped_train_dict = wrap_data_dict(handle_text(train_tweet_id2text), handle_text(train_tweet_id2issue), train_tweet_id2label)
    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_train_dict)
    text_to_embedding,issue_to_embedding = None, None


    num_of_labels = 17
    if(model_type == 'NN'):
        if(data_format == 'BoW'):
            num_of_each_x = len(text_word_dict_to_ix) + len(issue_word_dict_to_ix)
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            num_of_each_x = 3*embedding_size
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):
            num_of_each_x = (Max_padding_size+1)*embedding_size
        else:
            num_of_each_x = 2*embedding_size

    elif(model_type == 'NN2'):
        if(data_format == 'BoW'):
            num_of_each_x = len(text_word_dict_to_ix) + len(issue_word_dict_to_ix)
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            num_of_each_x = 3*embedding_size
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):
            num_of_each_x = (Max_padding_size+1)*embedding_size
        else:
            num_of_each_x = 2*embedding_size
        
    elif(model_type == 'CNN'):
        num_of_each_x = (Max_padding_size+1)*embedding_size
    else:
        print("No suitable size.")

    if(model_type == 'NN'):
        model = NeuralNet(num_of_each_x,num_of_labels,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    elif(model_type == 'NN2'):
        model = NeuralNet_2(num_of_each_x,num_of_labels,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    elif(model_type == 'CNN'):
        model = CNN(embedding_size, num_of_each_x, [3,4,5], num_of_labels, 0.5,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    else:
        print("Please set a feasible training model: NN, ...")

    if(data_format == 'BoW'):
        x_matrix, y_matrix, key_list, label_type_list = BoW(wrapped_train_dict,text_word_dict_to_ix,issue_word_dict_to_ix)
    elif(data_format == 'WordEmbedding'):
        x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
            Max_padding_size,model.text_embedding,model.issue_embedding)
    elif(data_format == 'NoFlattenEmbedding'):
        x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form_2d(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
            Max_padding_size,model.text_embedding,model.issue_embedding)
    else:
        print("Please set a feasible data pre-processing method: BoW, WordEmbedding, ...")

    train_x,train_y,validation_x,validation_y,test_x,test_y = partition(x_matrix,y_matrix,key_list,train_ratio,validation_ratio,test_ratio)
    model,train_accu_array,validation_accu_array,epoch_num_array = train(model,train_x,train_y,validation_x,validation_y,label_type_list,learning_rate,weight_decay_rate,number_of_epoches,
        embedding_size,whether_analyze,num_of_record_results,model_type,loss_function_choice,update_method,
    wrapped_train_dict,embedding_way_for_diff_length_sentence,Max_padding_size,model.text_embedding,model.issue_embedding)
    
    return model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding,train_x,train_y,validation_x,validation_y, \
        test_x,test_y,label_type_list,train_accu_array,validation_accu_array,wrapped_train_dict,model.text_embedding,model.issue_embedding, \
        epoch_num_array

def infer_unlabeled_data(model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding,
    data_folder_name,value_of_top_k,Max_padding_size):

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

    wrapped_dict = wrap_data_dict(handle_text(unlabeled_text), handle_text(unlabeled_issue), unlabeled_label)
    
    if(data_format == 'BoW'):
        x_matrix, y_matrix, key_list, label_type_list = BoW(wrapped_dict,text_word_dict_to_ix,issue_word_dict_to_ix)
    elif(data_format == 'WordEmbedding'):

        text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
        text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)
        issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)

        x_matrix,y_matrix = handle_data_with_word_embedding(text_embedding_list,issue_embedding_list,label_list,
            embedding_way_for_diff_length_sentence,Max_padding_size)

    elif(data_format == 'NoFlattenEmbedding'):
            
        text_list,issue_list,key_list,label_list,label_type_list = wrap_dict_to_separate_list(wrapped_dict)
        text_embedding_list = wrap_embedding_training_set(text_list,text_to_embedding)
        issue_embedding_list = wrap_embedding_training_set(issue_list,issue_to_embedding)

        x_matrix,y_matrix = handle_data_with_word_embedding_2d(text_embedding_list,issue_embedding_list,label_list,
            embedding_way_for_diff_length_sentence,Max_padding_size)
    else:
        print("Please set a feasible data pre-processing method: BoW, WordEmbedding, ...")

    print("Predicting their labels...")
    tensor_x_matrix = torch.FloatTensor(x_matrix)
    print("All loaded unlabeled data size:" + str(len(tensor_x_matrix)))

    k = value_of_top_k
    whether_use_top_k = 1
    if(whether_use_top_k):
        label_count = {}
        value_for_each_label = {}
        index_for_each_label = {}

        #select_x = torch.FloatTensor(np.zeros((1,len(tensor_x_matrix[0]))))
        select_x = torch.unsqueeze(torch.FloatTensor(np.zeros((tensor_x_matrix.shape[1:]))), 0)
        #print(tensor_x_matrix[0].shape)
        #print(select_x.shape)
        select_y = [0]

        select_key_list = [key_list[0]]

        for i in range(len(tensor_x_matrix)):
            cur_key = key_list[i]
            input_value = torch.unsqueeze(tensor_x_matrix[i], 0)
            result = torch.squeeze(model.forward(input_value))

            cur_predict_y = torch.argmax(result)
            cur_value = result[torch.argmax(result)]

            unlabeled_label[int(cur_key)] = int(cur_predict_y)

            if(int(cur_predict_y) not in label_count):
                label_count[int(cur_predict_y)] = 1
                value_for_each_label[int(cur_predict_y)] = torch.FloatTensor([cur_value])
                index_for_each_label[int(cur_predict_y)] = [len(select_x)]

                #print(select_x.shape)
                #print(tensor_x_matrix[i].shape)
                select_x = torch.cat((select_x,torch.unsqueeze(tensor_x_matrix[i], 0)),0)
                #select_x = torch.cat((select_x,tensor_x_matrix[i]),0)
                select_y.append(cur_predict_y)
                select_key_list.append(cur_key)
                #print(select_x.shape)
            else:
                if(label_count[int(cur_predict_y)] < k):
                    label_count[int(cur_predict_y)] += 1
                    value_for_each_label[int(cur_predict_y)]=torch.cat((value_for_each_label[int(cur_predict_y)],torch.Tensor([cur_value])),0)
                    index_for_each_label[int(cur_predict_y)].append(len(select_x))

                    select_x = torch.cat((select_x,torch.unsqueeze(tensor_x_matrix[i], 0)),0)
                    select_y.append(cur_predict_y)
                    select_key_list.append(cur_key)
                else:
                    cur_min_value_index_in_selected_y = torch.argmin(value_for_each_label[int(cur_predict_y)])
                    cur_min_value = value_for_each_label[int(cur_predict_y)][cur_min_value_index_in_selected_y]
                    corresponding_index_in_x = index_for_each_label[int(cur_predict_y)][cur_min_value_index_in_selected_y]
                    if(cur_value > cur_min_value):
                        value_for_each_label[int(cur_predict_y)][cur_min_value_index_in_selected_y] = torch.Tensor([cur_value])

                        #print(index_for_each_label)
                        #print(corresponding_index_in_x)
                        select_x[corresponding_index_in_x]=tensor_x_matrix[i]
                        select_y[corresponding_index_in_x]=cur_predict_y
                        select_key_list[corresponding_index_in_x]=cur_key

        #print(select_x)
        select_x = torch.FloatTensor(select_x)
        select_y = torch.LongTensor(select_y)

        print(select_x.shape)
        print(select_y.shape)
        print(label_count)
        return copied_text,unlabeled_issue,unlabeled_label,select_x,select_y,select_key_list
    else:

        for i in range(len(tensor_x_matrix)):
            cur_key = key_list[i]
            input_value = torch.unsqueeze(tensor_x_matrix[i], 0)
            cur_predict_y = torch.argmax(torch.squeeze(model.forward(input_value)))

            unlabeled_label[int(cur_key)] = int(cur_predict_y)
            y_matrix[i] = cur_predict_y
            tensor_y_matrix = torch.LongTensor(y_matrix)
        return copied_text,unlabeled_issue,unlabeled_label,tensor_x_matrix,tensor_y_matrix, key_list
    '''
    print(unlabeled_text)
    print(unlabeled_issue)
    print(unlabeled_label)
    print(len(unlabeled_text),len(unlabeled_issue),len(unlabeled_label))
    '''

def execute(data_format,model_type,train_ratio,validation_ratio,test_ratio,
    learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
    whether_analyze,num_of_record_results,image_folder,update_method,
    loss_function_choice,embedding_way_for_diff_length_sentence,
    data_folder_name,value_of_top_k,Max_padding_size):

    model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding, \
        train_x,train_y,validation_x,validation_y,test_x,test_y,label_type_list,first_train_accu_array, \
        first_validation_accu_array,wrapped_train_dict,text_embedding, \
        issue_embedding,epoch_num_array = first_train(data_format,model_type,train_ratio,validation_ratio,test_ratio,
        learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
        whether_analyze,num_of_record_results,image_folder,update_method,
        loss_function_choice,embedding_way_for_diff_length_sentence,Max_padding_size)


    first_test_accuracy = evaluate(model,test_x,test_y)

    print("Original training data size (D):" + str(len(train_x)))

    unlabeled_text,unlabeled_issue,unlabeled_label,extra_x,extra_y,select_key_list = infer_unlabeled_data(model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,
        issue_to_embedding,data_folder_name,value_of_top_k,Max_padding_size)

    #print(unlabeled_text,unlabeled_issue,unlabeled_label)

    new_text_dict = {}
    new_issue_dict = {}
    new_label_dict = {}

    for key in unlabeled_text:
        new_text_dict[key] = str(unlabeled_text[key])
        new_issue_dict[key] = str(unlabeled_issue[key])
        new_label_dict[key] = unlabeled_label[key]

    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2label = ReadFile('train.csv')

    for key in train_tweet_id2text:
        new_text_dict[key] = train_tweet_id2text[key]
        new_issue_dict[key] = train_tweet_id2issue[key]
        new_label_dict[key] = train_tweet_id2label[key]

    copied_train_text = train_tweet_id2text.copy()

    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2label = new_text_dict,new_issue_dict,new_label_dict
    '''
    Implement your Neural Network classifier here
    '''

    wrapped_train_dict = wrap_data_dict(handle_text(train_tweet_id2text), handle_text(train_tweet_id2issue), train_tweet_id2label)
    text_word_dict_to_ix,issue_word_dict_to_ix = get_all_dict_to_ix(wrapped_train_dict)
    text_to_embedding,issue_to_embedding = None, None



    num_of_labels = 17
    if(model_type == 'NN'):
        if(data_format == 'BoW'):
            num_of_each_x = len(text_word_dict_to_ix) + len(issue_word_dict_to_ix)
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            num_of_each_x = 3*embedding_size
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):
            num_of_each_x = (Max_padding_size+1)*embedding_size
        else:
            num_of_each_x = 2*embedding_size

    elif(model_type == 'NN2'):
        if(data_format == 'BoW'):
            num_of_each_x = len(text_word_dict_to_ix) + len(issue_word_dict_to_ix)
        elif(embedding_way_for_diff_length_sentence == 'MinMax'):
            num_of_each_x = 3*embedding_size
        elif(embedding_way_for_diff_length_sentence == 'PaddingByMax'):
            num_of_each_x = (Max_padding_size+1)*embedding_size
        else:
            num_of_each_x = 2*embedding_size
        
    elif(model_type == 'CNN'):
        num_of_each_x = (Max_padding_size+1)*embedding_size
    else:
        print("No suitable size.")

    if(model_type == 'NN'):
        new_model = NeuralNet(num_of_each_x,num_of_labels,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    elif(model_type == 'NN2'):
        new_model = NeuralNet_2(num_of_each_x,num_of_labels,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    elif(model_type == 'CNN'):
        new_model = CNN(embedding_size, num_of_each_x, [3,4,5], num_of_labels, 0.5,text_word_dict_to_ix,issue_word_dict_to_ix,embedding_size)
    else:
        print("Please set a feasible training model: NN, ...")

    if(data_format == 'BoW'):
        x_matrix, y_matrix, key_list, label_type_list = BoW(wrapped_train_dict,text_word_dict_to_ix,issue_word_dict_to_ix)
    elif(data_format == 'WordEmbedding'):
        x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
            Max_padding_size,new_model.text_embedding,new_model.issue_embedding)
    elif(data_format == 'NoFlattenEmbedding'):
        x_matrix, y_matrix, key_list, label_list, label_type_list,text_to_embedding,issue_to_embedding = get_embedding_form_2d(wrapped_train_dict,embedding_size,embedding_way_for_diff_length_sentence,
            Max_padding_size,new_model.text_embedding,new_model.issue_embedding)
    else:
        print("Please set a feasible data pre-processing method: BoW, WordEmbedding, ...")

    print("Combined size:" + str(len(x_matrix)))
    print("Get new model based on D and U.")

    train_x,train_y,validation_x,validation_y,test_x,test_y = partition(x_matrix,y_matrix,key_list,train_ratio,validation_ratio,test_ratio)
    new_model,train_accu_array,validation_accu_array,epoch_num_array = train(new_model,train_x,train_y,validation_x,validation_y,label_type_list,learning_rate,weight_decay_rate,number_of_epoches,
        embedding_size,whether_analyze,num_of_record_results,model_type,loss_function_choice,update_method,
    wrapped_train_dict,embedding_way_for_diff_length_sentence,Max_padding_size,new_model.text_embedding,new_model.issue_embedding)
    

    print("First Test accuracy: " + str(first_test_accuracy))

    new_test_accuracy = evaluate(new_model,test_x,test_y)

    print("New Test accuracy: " + str(new_test_accuracy))


    plt.plot(epoch_num_array, first_train_accu_array, label='first_train_accu_array')
    plt.plot(epoch_num_array, first_validation_accu_array, label='first_validation_accu_array')
    plt.plot(epoch_num_array, train_accu_array, label='new_train_accu_array')
    plt.plot(epoch_num_array, validation_accu_array, label='new_validation_accu_array')
    plt.xlabel('epoch_num_array')
    plt.ylabel('accuracy')
    plt.legend()
    #plt.savefig('F1score_diff_lr_op'+str(args.option)+'_sgd.png')

    image_file_name = image_folder + '/' + data_format + '_' + model_type \
        + '_' + update_method + '_' + 'lr' + str(learning_rate) + '_' + 'epoch' \
        + str(number_of_epoches) + '.png'
    if(data_format == 'WordEmbedding' or data_format == 'NoFlattenEmbedding'):
        image_file_name += '_' + embedding_way_for_diff_length_sentence + \
            '_topk' + str(value_of_top_k) + '.png'
    print(image_file_name)
    plt.savefig(image_file_name)
    plt.close()

    return


if __name__ == '__main__':

    train_ratio = 0.7
    validation_ratio = 0.2
    test_ratio = 0.1

    learning_rate = 0.0005
    weight_decay_rate = 1e-4
    number_of_epoches = 50#300
    whether_analyze = 1 #1:yes,0:no
    num_of_record_results = 50
    embedding_size = 1000
    issue_embedding_size = embedding_size
    value_of_top_k = 100
    Max_padding_size = 20
    hidden_dim = 256
    use_pretrained = False
    compare_two_models = True
    semi_supervised = False
    vocab_file = 'word_to_embedding.pkl'

    #data format can be chosen from: BoW, WordEmbedding,NoFlattenEmbedding
    data_format = 'BoW'
    #embedding_way_for_diff_length_sentence can be chosen from: 
    #Average, Sum, Min, Max, MinMax, PaddingByMax
    embedding_way_for_diff_length_sentence = 'PaddingByMax'

    #option: NN, NN2, CNN...
    model_type = 'NN'
    loss_function_choice = 'NLLLoss'

    #update method can be chosen from: SGD, Adam
    update_method = 'Adam'

    data_folder_name = 'data'
    image_folder = 'result'#'final_semi'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if(compare_two_models == False and semi_supervised == False):
        #simply do training and prediction once

        print("NN itself")

        model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding, \
            train_x,train_y,validation_x,validation_y,test_x,test_y,label_type_list,first_train_accu_array, \
            first_validation_accu_array,wrapped_train_dict,text_embedding, \
            issue_embedding,first_epoch_num_array = first_train(data_format,model_type,train_ratio,validation_ratio,test_ratio,
            learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
            whether_analyze,num_of_record_results,image_folder,update_method,
            loss_function_choice,embedding_way_for_diff_length_sentence,Max_padding_size)

        test_accuracy = evaluate(model,test_x,test_y)

        print("New Test accuracy: " + str(test_accuracy))

    if(compare_two_models == True):

        print("NN vs CNN")

        model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding, \
            train_x,train_y,validation_x,validation_y,test_x,test_y,label_type_list,first_train_accu_array, \
            first_validation_accu_array,wrapped_train_dict,text_embedding, \
            issue_embedding,first_epoch_num_array = first_train(data_format,model_type,train_ratio,validation_ratio,test_ratio,
            learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
            whether_analyze,num_of_record_results,image_folder,update_method,
            loss_function_choice,embedding_way_for_diff_length_sentence,Max_padding_size)

        first_test_accuracy = evaluate(model,test_x,test_y)


        data_format = 'WordEmbedding'
        embedding_way_for_diff_length_sentence = 'PaddingByMax'
        model_type = 'NN'
        use_pretrained = False

        model,text_word_dict_to_ix,issue_word_dict_to_ix,text_to_embedding,issue_to_embedding, \
            train_x,train_y,validation_x,validation_y,test_x,test_y,label_type_list,second_train_accu_array, \
            second_validation_accu_array,wrapped_train_dict,text_embedding, \
            issue_embedding,second_epoch_num_array = first_train(data_format,model_type,train_ratio,validation_ratio,test_ratio,
            learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
            whether_analyze,num_of_record_results,image_folder,update_method,
            loss_function_choice,embedding_way_for_diff_length_sentence,Max_padding_size)
        


        print("First Test accuracy: " + str(first_test_accuracy))

        new_test_accuracy = evaluate(model,test_x,test_y)

        print("New Test accuracy: " + str(new_test_accuracy))

        plt.plot(first_epoch_num_array, first_train_accu_array, label='Train accuracy (baseline)')
        plt.plot(first_epoch_num_array, first_validation_accu_array, label='Validation accuracy (baseline)')
        plt.plot(first_epoch_num_array, second_train_accu_array, label='Train accuracy (extension)')
        plt.plot(first_epoch_num_array, second_validation_accu_array, label='Validation accuracy (extension)')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.savefig('F1score_diff_lr_op'+str(args.option)+'_sgd.png')
        plt.savefig('compare_two_models.png')
        plt.close()

    if(semi_supervised == True):
        execute(data_format,model_type,train_ratio,validation_ratio,test_ratio,
            learning_rate,weight_decay_rate,number_of_epoches,embedding_size,
            whether_analyze,num_of_record_results,image_folder,update_method,
            loss_function_choice,embedding_way_for_diff_length_sentence,
            data_folder_name,value_of_top_k,Max_padding_size)




