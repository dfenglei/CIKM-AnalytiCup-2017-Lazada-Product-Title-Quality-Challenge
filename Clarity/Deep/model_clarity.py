#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Last Modified on Wed Aug 16 16:35 2017

@author: Vishal
"""

import numpy as np
import timeit
import argparse
import keras
import pdb
import csv
import cPickle as cpkl
import pylab as pl
from pylab import *
import matplotlib.pyplot as plt
import string
import scipy
import re
from keras.layers.normalization import BatchNormalization
from collections import defaultdict
from sklearn import metrics
from keras import regularizers
from keras import backend as K
from keras.utils import plot_model
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D
from keras.datasets import imdb
from keras.layers.core import Reshape
from keras.layers.merge import Dot,Concatenate
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

np.random.seed(1337) # for reproducibility

#=========================================================================================
TRAIN_DATA_PATH = 'data/train.csv'
VALID_DATA_PATH = 'data/valid.csv'
TEST_DATA_PATH = 'data/test.csv'
VOCAB_PATH = 'Vocab2Vec/Vocab2Vec.p'
LOG_PATH = "logs/"
OUTPUT = 'clarity_test.predict'
#=========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='whether to train')
parser.add_argument('--validate', action='store_true', help='whether to validate')
parser.add_argument('--test', action='store_true', help='whether to test')
parser.add_argument('--batchsize', default=128)
parser.add_argument('--learningrate', default=0.0001)
parser.add_argument('--weightspath', default='weights/weights.hdf5')
args = parser.parse_args()
#=========================================================================================
WEIGHTS_PATH_BEST = args.weightspath
TRAIN = args.train
GET_TEST_RESULTS = args.test
GET_VALIDATION_RMSE = args.validate
LOAD_BEST_WTS = False
NUM_FEAT = 267 #280 284 267 271
MAX_TITLE_LEN = 45
MAX_CAT_LEN = 10
EMB_DIM = 300
EPOCHS = 40
BATCH_SIZE = int(args.batchsize)
LEARNING_RATE = float(args.learningrate)
#=========================================================================================
sw		=	stopwords.words('english')
exclude 	= 	set(string.punctuation)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
#=========================================================================================
def get_difference(vects):
	x,y=vects
	return x-y


def get_diff_shape(shapes):
	shape1, shape2 = shapes
	return shape1


def get_transpose(x):
	return K.permute_dimensions(x,(0,2,1))


def transpose_shape(shape):
	return (shape[0],shape[2],shape[1])


def get_matmul(mats):
	x,y = mats
	return K.batch_dot(x,y)


def matmul_shape(shapes):
	shape1,shape2=shapes
	return (shape1[0],shape1[1],shape2[2])


def remove_nums(word):

	return ''.join([i for i in word if not i.isdigit()])


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
	xt		=	[word for word in x]# if word not in sw]
	return xt


def get_text2wvecs(text,exclude,vocab2vec,maxlen):
	
	words = preprocess_line(text,exclude)
	vocab = set(words)
	li=[]
	oov = set()
	for word in vocab:
		if word in vocab2vec:
			li.append(vocab2vec[word])
		else:
			oov.add(word)
	if len(li)<maxlen:
		for ii in range(maxlen-len(li)):
			ze = np.zeros((300,))
			li.append(ze)
	else:
		li = li[0:maxlen]
	return np.transpose(np.asarray(li))



def rmse(true,pred):
	error=0
	for i,val in enumerate(true):
		error+=(val[1]-pred[i][1])**2
	error = error/true.shape[0]
	return np.sqrt(error)
	

vocab2vec=cpkl.load(open(VOCAB_PATH,'r'))


data2use_train_feat = csv.reader(open(TRAIN_DATA_PATH,'r'))
data2use_valid_feat = csv.reader(open(VALID_DATA_PATH,'r'))
data2use_test_feat = csv.reader(open(TEST_DATA_PATH,'r'))
data2use_train = map(list,zip(*list(csv.reader(open(TRAIN_DATA_PATH,'r')))))
data2use_valid = map(list,zip(*list(csv.reader(open(VALID_DATA_PATH,'r')))))
data2use_test = map(list,zip(*list(csv.reader(open(TEST_DATA_PATH,'r')))))


print 'Preparing Data ...'
start_time = timeit.default_timer()
if TRAIN:
	title_train		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_TITLE_LEN) for line in data2use_train[-4]])
	print 'Shape of Title for training = ',title_train.shape
	cat_train		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_train[-3]])
	print 'Shape of Category for training = ',cat_train.shape
	label_tmp_train		=	np.asarray([int(line) for line in data2use_train[-2]])
	label_train_i		=	np.reshape(label_tmp_train,(label_tmp_train.shape[0],1))
	label_train		=	np_utils.to_categorical(label_train_i)
	print label_train.shape
	features_train		=	np.asarray([line[0:-4] for line in data2use_train_feat])
	print 'Shape of Features for training = ',features_train.shape

if TRAIN or GET_VALIDATION_RMSE:
	title_valid		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_TITLE_LEN) for line in data2use_valid[-4]])
	print 'Shape of Title for validation = ',title_valid.shape
	cat_valid		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_valid[-3]])
	print 'Shape of Category for validation = ',cat_valid.shape
	label_tmp_valid		=	np.asarray([int(line) for line in data2use_valid[-2]])
	label_valid_i		=	np.reshape(label_tmp_valid,(label_tmp_valid.shape[0],1))
	label_valid		=	np_utils.to_categorical(label_valid_i)
	print label_valid.shape
	features_valid		=	np.asarray([line[0:-4] for line in data2use_valid_feat])
	print 'Shape of Features for validation = ',features_valid.shape

if GET_TEST_RESULTS:
	title_test		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_TITLE_LEN) for line in data2use_test[-2]])
	print 'Shape of Title for testing = ',title_test.shape
	cat_test		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_test[-1]])
	print 'Shape of Category for testing = ',cat_test.shape
	features_test		=	np.asarray([line[0:-2] for line in data2use_test_feat])
	print 'Shape of Features for testing = ',features_test.shape
print 'Done!'
elapsed = timeit.default_timer() - start_time
print 'Data creation took ',elapsed/60.0,' minutes ...'

####APonW2V####
T=Input(shape=(EMB_DIM,MAX_TITLE_LEN,), dtype='float32') #300xT  Title word2vec
C=Input(shape=(EMB_DIM,MAX_CAT_LEN,), dtype='float32') #300xC  Category word2vec
FEAT=Input(shape=(NUM_FEAT,), dtype='float32') # Generated features
FEAT_R=Reshape((1,NUM_FEAT))(FEAT)
T_T = Lambda(get_transpose, output_shape=transpose_shape)(T)   #Tx300
C_T = Lambda(get_transpose, output_shape=transpose_shape)(C)   #Cx300
C_TU = Dense(EMB_DIM,use_bias=False,kernel_initializer='Identity')(C_T)  #Cx300
G_1 =  Lambda(get_matmul, output_shape=matmul_shape)([C_TU, T]) #CxT
G = Activation(K.tanh)(G_1) #CxT
G_T =  Lambda(get_transpose, output_shape=transpose_shape)(G) #TxC
T_maxpool = MaxPooling1D(pool_size=MAX_CAT_LEN, strides=None, padding='valid')(G) #1xT
C_maxpool = MaxPooling1D(pool_size=MAX_TITLE_LEN, strides=None, padding='valid')(G_T) #1xC
T_soft = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(T_maxpool)) #Tx1
C_soft = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(C_maxpool)) #Cx1
outt = Lambda(get_matmul, output_shape=matmul_shape)([T, T_soft]) #300x1
outt_T = Lambda(get_transpose, output_shape=transpose_shape)(outt) #1x300
outc = Lambda(get_matmul, output_shape=matmul_shape)([C, C_soft]) #300x1
outc_T = Lambda(get_transpose, output_shape=transpose_shape)(outc)#1x300
FINAL_EMB = Dense(EMB_DIM,use_bias=False,kernel_initializer='Identity')    
outtv = FINAL_EMB(outt_T)
outcv = FINAL_EMB(outc_T)
out_diff = Lambda(get_difference, output_shape=get_diff_shape)([outtv,outcv])
###APonCNN###
t_reshape_layer			= 	Reshape((EMB_DIM,MAX_TITLE_LEN,1))
c_reshape_layer			= 	Reshape((EMB_DIM,MAX_CAT_LEN,1))
conv_layer			= 	Conv2D(100, (300,3), strides=(1, 1),data_format='channels_last',activation='tanh')
pooling_layer_t			=	MaxPooling2D(pool_size=(1,MAX_TITLE_LEN-2))
pooling_layer_c			=	MaxPooling2D(pool_size=(1,MAX_CAT_LEN-2))
final_reshape_layer		= 	Reshape((1,100))
t_mat = Reshape((MAX_TITLE_LEN-2,100))(conv_layer(t_reshape_layer(T))) #Tx100
t_mat_T = Lambda(get_transpose, output_shape=transpose_shape)(t_mat)   #100xT
c_mat = Reshape((MAX_CAT_LEN-2,100))(conv_layer(c_reshape_layer(C))) #Cx100
c_mat_T = Lambda(get_transpose, output_shape=transpose_shape)(c_mat)   #100xC
c_matV = Dense(100,use_bias=False,kernel_initializer='Identity')(c_mat)  #Cx100
G_2 =  Lambda(get_matmul, output_shape=matmul_shape)([c_matV, t_mat_T]) #CxT
G_act = Activation(K.tanh)(G_2) #CxT
G_act_T =  Lambda(get_transpose, output_shape=transpose_shape)(G_act) #TxC
T_maxpool_2 = MaxPooling1D(pool_size=MAX_CAT_LEN-2, strides=None, padding='valid')(G_act) #1xT
C_maxpool_2 = MaxPooling1D(pool_size=MAX_TITLE_LEN-2, strides=None, padding='valid')(G_act_T) #1xC
T_soft_2 = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(T_maxpool_2)) #Tx1
C_soft_2 = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(C_maxpool_2)) #Cx1
outt2 = Lambda(get_matmul, output_shape=matmul_shape)([t_mat_T, T_soft_2]) #100x1
out_tconv = Lambda(get_transpose, output_shape=transpose_shape)(outt2) #1x100
outc2 = Lambda(get_matmul, output_shape=matmul_shape)([c_mat_T, C_soft_2]) #100x1
out_cconv =Lambda(get_transpose, output_shape=transpose_shape)(outc2) #1x100
FINAL_EMB2 = Dense(100,use_bias=False,kernel_initializer='Identity')
out_tconv2 = FINAL_EMB2(out_tconv)
out_cconv2 = FINAL_EMB2(out_cconv)
####OUTPUT####
out_conc = Concatenate(axis=2)([outtv,outcv,out_diff,out_tconv2,out_cconv2,FEAT_R])
out_layer_1 = Dense(10,activation='relu')(out_conc)
out_layer_2 = Dense(10,activation='relu')(out_layer_1)
out_layer_3 = Dense(10,activation='relu')(out_layer_2)
out_layer_4 = Dense(10,activation='relu')(out_layer_3)
out_layer_5 = Dense(10,activation='relu')(out_layer_4)
out_pen = Dense(2,activation='softmax')(out_layer_5)
out_final = Reshape((2,))(out_pen)
model = Model(inputs=[T, C, FEAT], outputs=out_final)

opt = keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=8.0, clipnorm=5.0)

if LOAD_BEST_WTS or GET_TEST_RESULTS or (not TRAIN and GET_VALIDATION_RMSE):
	model.load_weights(WEIGHTS_PATH_BEST)

model.compile(optimizer=opt, loss=losses.categorical_crossentropy)


if TRAIN:
	checkpoint1 = ModelCheckpoint(WEIGHTS_PATH_BEST, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
	checkpoint2 = keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)  #Give log path to see progress on tensorboard
	callbacks_list = [checkpoint1,checkpoint2]
	print 'training .....'
	start_time = timeit.default_timer()
	model.fit([title_train, cat_train, features_train], label_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=([title_valid, cat_valid, features_valid], label_valid),shuffle=True,callbacks=callbacks_list)
	elapsed = timeit.default_timer() - start_time
	print 'Training took ',elapsed/60.0,' minutes'


if GET_VALIDATION_RMSE:	  #To get Validation rmse on holdout set
	output = model.predict([title_valid, cat_valid, features_valid])
	print 'Holdout set RMSE = ', rmse(label_valid,output)


if GET_TEST_RESULTS:	#To get predictions on test set
	output = model.predict([title_test, cat_test, features_test])
	fw = open(OUTPUT,'w')
	for elem in output:
		fw.write(str(elem[1])+'\n')
	fw.close()
	print 'Done predictions!'
