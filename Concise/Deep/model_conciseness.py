#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Last Modified on Wed Aug 16 15:40 2017

@author: Vishal
"""

import timeit
import argparse
import numpy as np
import pdb
import csv
import cPickle as cpkl
import keras
import string
import scipy
import re
import pandas as pd
from scipy import spatial
from pyjarowinkler import distance
from sklearn import metrics
from keras import regularizers
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Dropout,Input,Lambda,Embedding,LSTM,Activation,Conv2D,Conv1D,MaxPooling2D,MaxPooling1D
from keras.models import Model,Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Reshape,Flatten
from keras.layers.merge import Dot,Concatenate
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

np.random.seed(1337)  #for reproducibility 6042

#=========================================================================================
TRAIN_DATA_PATH = 'data/train.csv'
VALID_DATA_PATH = 'data/valid.csv'
TEST_DATA_PATH = 'data/test.csv'
MAT_PATH_TRAIN = 'matrices/mat4title_train.p'
MAT_PATH_VALID = 'matrices/mat4title_valid.p'
MAT_PATH_TEST = 'matrices/mat4title_test.p'
VOCAB_PATH = 'Vocab2Vec/Vocab2Vec.p'     
LOG_PATH = 'logs/'
OUTPUT = 'conciseness_test.predict'
#=========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='whether to train')
parser.add_argument('--validate', action='store_true', help='whether to validate')
parser.add_argument('--test', action='store_true', help='whether to test')
parser.add_argument('--mats', action='store_true', help='if true->get syn/sem mats from scratch else load precomputed')
parser.add_argument('--batchsize', default=128)
parser.add_argument('--learningrate', default=0.0001)
parser.add_argument('--weightspath', default='weights/weights.hdf5')
args = parser.parse_args()
#=========================================================================================
WEIGHTS_PATH_BEST = args.weightspath
TRAIN = args.train
GET_TEST_RESULTS = args.test
GET_VALIDATION_RMSE = args.validate
GET_MATS = args.mats
LOAD_BEST_WTS = False
NUM_FEAT = 288 #280 267
MAX_TITLE_LEN = 45
MAX_RAW_TITLE_LEN = 50
EMB_DIM = 300
EPOCHS = 30
BATCH_SIZE = int(args.batchsize)
LEARNING_RATE = float(args.learningrate)
#=========================================================================================
sw		=	stopwords.words('english')
exclude 	= 	set(string.punctuation)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
#=========================================================================================
def remove_nums(word):

	return ''.join([i for i in word if not i.isdigit()])


def get_sequences(sentences, word2id, vocab):
	
	sequences = []
	for sentence in sentences:
		words = word_tokenize(sentence)
		seq = []
		for word in words:
			if word in vocab:
				seq.append(word2id[word])
			else:
				seq.append(word2id['unk'])
		sequences.append(seq)
	return sequences


def get_sequence_data(data, word2id, vocab):
	
	sequences = get_sequences(data, word2id, vocab)
	padded_seqs = pad_sequences(sequences, maxlen=MAX_TITLE_LEN, dtype='int32', padding='post', truncating='post', value=0)
	return padded_seqs


def get_sequence_data_2(data, word2id, vocab):
	
	sequences = get_sequences(data, word2id, vocab)
	padded_seqs = pad_sequences(sequences, maxlen=MAX_CAT_LEN, dtype='int32', padding='post', truncating='post', value=0)
	return padded_seqs


def preprocess_line(line,exclude):

	st		=	(line.strip()).lower()
	st		=	re.sub(r'[^\x00-\x7F]+',' ', st)
	words		=	word_tokenize(st)
	new		=	[]
	for i in range(len(words)):
		if words[i] not in exclude:
			new.append(remove_nums(words[i]))
	#newl		=	[word for word in new if word not in sw]	
	new_line	=	' '.join(new)
	line_wo_pun	=	new_line.translate(replace_punctuation)
	x		=	word_tokenize(line_wo_pun)
	xt		=	[word for word in x if len(word)>1]
	return ' '.join(xt)


def preprocess_line_2(line,exclude):

	st		=	(line.strip()).lower()
	st		=	re.sub(r'[^\x00-\x7F]+',' ', st)
	words		=	word_tokenize(st)
	new		=	[]
	for i in range(len(words)):
		if words[i] not in exclude:
			new.append(remove_nums(words[i]))
	#newl		=	[word for word in new if word not in sw]	
	new_line	=	' '.join(new)
	line_wo_pun	=	new_line.translate(replace_punctuation)
	x		=	word_tokenize(line_wo_pun)
	xt		=	[word for word in x]# if word not in sw]
	return xt


def preprocess_line_syn(line,exclude):

	st		=	(line.strip()).lower()
	st		=	re.sub(r'[^\x00-\x7F]+',' ', st)
	words		=	word_tokenize(st)
	new		=	[]
	for i in range(len(words)):
		if words[i] not in exclude:
			new.append(words[i])
	#newl		=	[word for word in new if word not in sw]	
	new_line	=	' '.join(new)
	line_wo_pun	=	new_line.translate(replace_punctuation)
	x		=	word_tokenize(line_wo_pun)
	xt		=	[word for word in x]# if word not in sw]
	return xt


def get_synmat4title(title,maxlen):
	
	splitTitle = preprocess_line_syn(title,exclude)
	List1 = splitTitle
	List2 = splitTitle
	Matrix = np.zeros((maxlen,maxlen),dtype=np.float)
	if len(List1)<maxlen:
		for i in range(0,len(List1)):
			for j in range(0,len(List2)):
				Matrix[i,j] = distance.get_jaro_distance(List1[i],List2[j])
	else:
		for i in range(0,maxlen):
				for j in range(0,maxlen):
					Matrix[i,j] = distance.get_jaro_distance(List1[i],List2[j])
	return Matrix


def get_semmat4title(title,vocab2vec,maxlen):

	splitTitle = preprocess_line_2(title,exclude)
	List = splitTitle
	veclist = []
	for word in List:
		if word in vocab2vec:
			veclist.append(vocab2vec[word])
	List1 = veclist
	List2 = veclist
	Matrix = np.zeros((maxlen,maxlen),dtype=np.float)
	if len(List1)<maxlen:
		for i in range(0,len(List1)):
			for j in range(0,len(List2)):
				Matrix[i,j] = 1.0-spatial.distance.cosine(List1[i],List2[j])
	else:
		for i in range(0,maxlen):
				for j in range(0,maxlen):
					Matrix[i,j] = 1.0-spatial.distance.cosine(List1[i],List2[j])
	return Matrix


def rmse(true,pred):
	error=0
	for i,val in enumerate(true):
		error+=(val[0]-pred[i][0])**2
	error = error/true.shape[0]
	return np.sqrt(error)
	

vocab2vec=cpkl.load(open(VOCAB_PATH,'r'))


data2use_train_feat = csv.reader(open(TRAIN_DATA_PATH,'r'))
data2use_valid_feat = csv.reader(open(VALID_DATA_PATH,'r'))
data2use_test_feat = csv.reader(open(TEST_DATA_PATH,'r'))
data2use_train = map(list,zip(*list(csv.reader(open(TRAIN_DATA_PATH,'r')))))
data2use_valid = map(list,zip(*list(csv.reader(open(VALID_DATA_PATH,'r')))))
data2use_test = map(list,zip(*list(csv.reader(open(TEST_DATA_PATH,'r')))))


print 'Preparing word indexing and pretrained embedding matrix ...'
with open(VOCAB_PATH,'r') as f:
	w2v_dic = cpkl.load(f)
	vocab = set()
	word2id = {}
	id2word = {}
	for ii,word in enumerate(w2v_dic):
		word2id[word] = ii
		id2word[ii] = word
		vocab.add(word)
	embedding_matrix = np.zeros((len(w2v_dic),300))
	for i in range(len(w2v_dic)):
		embedding_matrix[i] = w2v_dic[id2word[i]]
	emvoc = {'word2id': word2id, 'id2word': id2word, 'vocab': vocab, 'embedding_matrix': embedding_matrix}
	VOCAB_SIZE = len(emvoc['vocab'])
	embedding_matrix = emvoc['embedding_matrix']
print 'Done!'


print 'Preparing Data ...'
start_time = timeit.default_timer()
if TRAIN:
	title_train		=	get_sequence_data([preprocess_line(line,exclude) for line in data2use_train[-4]], emvoc['word2id'], emvoc['vocab'])
	print 'Shape of Title for training = ',title_train.shape
	label_tmp_train		=	np.asarray([int(line) for line in data2use_train[-1]])
	label_train		=	np.reshape(label_tmp_train,(label_tmp_train.shape[0],1))
	features_train		=	np.asarray([line[0:-4] for line in data2use_train_feat]) #-4
	print 'Shape of Features for training = ',features_train.shape

	print 'Preparing SYN and SEM matrix for training ... this takes time!!'
	if GET_MATS:
		title_semmat_train = np.asarray([get_semmat4title(title,vocab2vec,MAX_TITLE_LEN) for title in data2use_train[-4]])
		title_synmat_train = np.asarray([get_synmat4title(title,MAX_RAW_TITLE_LEN) for title in data2use_train[-4]])
		mat_final_train = (title_synmat_train,title_semmat_train)
		cpkl.dump(mat_final_train,open(MAT_PATH_TRAIN,'w'))
	else:
		mat_final_train		=	cpkl.load(open(MAT_PATH_TRAIN,'r'))
		title_synmat_train,title_semmat_train	= mat_final_train[0],mat_final_train[1]
	print '		Shape of SYN matrix = ',title_synmat_train.shape
	print '		Shape of SEM matrix = ',title_semmat_train.shape
		

if TRAIN or GET_VALIDATION_RMSE:
	title_valid		=	get_sequence_data([preprocess_line(line,exclude) for line in data2use_valid[-4]], emvoc['word2id'], emvoc['vocab'])
	print 'Shape of Title for validation = ',title_valid.shape
	label_tmp_valid		=	np.asarray([int(line) for line in data2use_valid[-1]])
	label_valid		=	np.reshape(label_tmp_valid,(label_tmp_valid.shape[0],1))
	features_valid		=	np.asarray([line[0:-4] for line in data2use_valid_feat]) #-4
	print 'Shape of Features for validation = ',features_valid.shape

	print 'Preparing SYN and SEM matrix for validation ... this takes time!!'	
	if GET_MATS:
		title_semmat_valid = np.asarray([get_semmat4title(title,vocab2vec,MAX_TITLE_LEN) for title in data2use_valid[-4]])
		title_synmat_valid = np.asarray([get_synmat4title(title,MAX_RAW_TITLE_LEN) for title in data2use_valid[-4]])
		mat_final_valid = (title_synmat_valid,title_semmat_valid)
		cpkl.dump(mat_final_valid,open(MAT_PATH_VALID,'w'))
	else:
		mat_final_valid		=	cpkl.load(open(MAT_PATH_VALID,'r'))
		title_synmat_valid,title_semmat_valid	= mat_final_valid[0],mat_final_valid[1]
	print '		Shape of SYN matrix = ',title_synmat_valid.shape
	print '		Shape of SEM matrix = ',title_semmat_valid.shape


if GET_TEST_RESULTS:
	print 'Preparing SYN and SEM matrix for testing ... this takes time!!'
	if GET_MATS:
		title_semmat_test = np.asarray([get_semmat4title(title,vocab2vec,MAX_TITLE_LEN) for title in data2use_test[-2]])
		title_synmat_test = np.asarray([get_synmat4title(title,MAX_RAW_TITLE_LEN) for title in data2use_test[-2]])
		mat_final_test = (title_synmat_test,title_semmat_test)
		cpkl.dump(mat_final_test,open(MAT_PATH_TEST,'w'))
	else:
		mat_final_test		=	cpkl.load(open(MAT_PATH_TEST,'r'))
		title_synmat_test,title_semmat_test	= mat_final_test[0],mat_final_test[1]
	print '		Shape of SYN matrix = ',title_synmat_test.shape
	print '		Shape of SEM matrix = ',title_semmat_test.shape
	title_test		=	get_sequence_data([preprocess_line(line,exclude) for line in data2use_test[-2]], emvoc['word2id'], emvoc['vocab'])
	print 'Shape of Title for testing = ',title_test.shape
	features_test		=	np.asarray([line[0:-2] for line in data2use_test_feat])
	print 'Shape of Features for testing = ',features_test.shape 
print 'Done!'
elapsed = timeit.default_timer() - start_time
print 'Data creation took ',elapsed/60.0,' minutes ...'

####MODEL####
T=Input(shape=(MAX_TITLE_LEN,), dtype='int32')			#Title word2vec
SYN = Input(shape=(MAX_RAW_TITLE_LEN,MAX_RAW_TITLE_LEN), dtype='float32')			#Syntactic matrix
SEM = Input(shape=(MAX_TITLE_LEN,MAX_TITLE_LEN), dtype='float32')				#Semantic matrix
FEAT=Input(shape=(NUM_FEAT,), dtype='float32')			#Generated features
FEAT_R=Reshape((1,NUM_FEAT))(FEAT)
embedding_layer = Embedding(VOCAB_SIZE,EMB_DIM,weights=[embedding_matrix],trainable=True)		#Embedding layer for fine tuning pretrained word2vec
embedding_layer_output_T = Dropout(0.5)(embedding_layer(T))
####CNN_LSTM####
conv_layer_output = Conv1D(filters=32, kernel_size=3, activation='tanh')(embedding_layer_output_T)
pooling_layer_output = Dropout(0.2)(MaxPooling1D(pool_size=2)(conv_layer_output))
lstm_layer_output = Reshape((1,128))(Dropout(0.2)(LSTM(128)(pooling_layer_output)))
####CNN_MAT####
conv_syn_output = Conv1D(filters=128, kernel_size=3, activation='tanh')(SYN)
pooling_layer_syn_output = Reshape((1,128))(Dropout(0.2)(MaxPooling1D(pool_size=MAX_RAW_TITLE_LEN-2)(conv_syn_output)))
conv_sem_output = Conv1D(filters=128, kernel_size=3, activation='tanh')(SEM)
pooling_layer_sem_output = Reshape((1,128))(Dropout(0.2)(MaxPooling1D(pool_size=MAX_TITLE_LEN-2)(conv_sem_output)))
####OUTPUT####
out_conc = Concatenate(axis=2)([lstm_layer_output,pooling_layer_syn_output,pooling_layer_sem_output,FEAT_R])
out_layer_1 = Dense(50,activation='relu')(out_conc)
out_layer_2 = Dense(50,activation='relu')(out_layer_1)
out_layer_3 = Dense(50,activation='relu')(out_layer_2)
out_pen = Dense(1,activation='sigmoid')(out_layer_3)
out_final = Reshape((1,))(out_pen)
model = Model(inputs=[T,SYN,SEM,FEAT], outputs=out_final)

opt = keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=8.0, clipnorm=None)

if LOAD_BEST_WTS or GET_TEST_RESULTS or (not TRAIN and GET_VALIDATION_RMSE):
	model.load_weights(WEIGHTS_PATH_BEST)

model.compile(optimizer=opt, loss=losses.binary_crossentropy)


if TRAIN:
	checkpoint1 = ModelCheckpoint(WEIGHTS_PATH_BEST, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
	checkpoint2 = keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)  #Give log path to see progress on tensorboard
	callbacks_list = [checkpoint1,checkpoint2]
	print 'training .....'
	start_time = timeit.default_timer()
	model.fit([title_train,title_synmat_train,title_semmat_train,features_train], label_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=([title_valid,title_synmat_valid,title_semmat_valid,features_valid], label_valid),shuffle=True,callbacks=callbacks_list,class_weight=None)
	elapsed = timeit.default_timer() - start_time
	print 'Training took ',elapsed/60.0,' minutes'
	

if GET_VALIDATION_RMSE:	  #To get Validation rmse on holdout set
	output = model.predict([title_valid,title_synmat_valid, title_semmat_valid, features_valid])
	print 'Holdout set RMSE = ', rmse(label_valid,output)
	

if GET_TEST_RESULTS:	#To get predictions on test set
	output = model.predict([title_test,title_synmat_test,title_semmat_test,features_test])
	fw = open(OUTPUT,'w')
	for elem in output:
		fw.write(str(elem[0])+'\n')
	fw.close()
	print 'Done predictions!'
