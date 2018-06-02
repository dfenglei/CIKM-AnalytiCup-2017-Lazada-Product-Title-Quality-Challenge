import numpy as np
import timeit
import pandas as pd
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
from scipy import spatial
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
TRAIN_DATA_PATH = './../Input/data_train.csv'
TEST_DATA_PATH = './../Input/data_test.csv'
OUT_PATH_TR = './../OutPut/Train_Data-features-6.csv'
OUT_PATH_TE = './../OutPut/Test_Data-features-6.csv'
#=========================================================================================
MAX_TITLE_LEN = 45
MAX_CAT_LEN = 10
#=========================================================================================
sw		=	stopwords.words('english')
exclude 	= 	set(string.punctuation)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
#=========================================================================================
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


def get_text2wvecs(text,exclude,vocab2vec,maxlen,test=False):
	
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
	return transpose(np.asarray(li))


def get4feat(emb_title,emb_cat,title, cat_vec, OUT_PATH):

	title1 = emb_title.predict([title, cat_vec[0]])
	cat1 = emb_cat.predict([title, cat_vec[0]])
	title2 = emb_title.predict([title, cat_vec[1]])
	cat2 = emb_cat.predict([title, cat_vec[1]])
	title3 = emb_title.predict([title, cat_vec[2]])
	cat3 = emb_cat.predict([title, cat_vec[2]])
	title4 = emb_title.predict([title, cat_vec[3]])
	cat4 = emb_cat.predict([title, cat_vec[3]])
	score1=[]
	for ii,elem in enumerate(title1):
		#print np.linalg.norm(elem-cat[ii])
		dist = 1 - spatial.distance.cosine(elem, cat1[ii])
		
		if not isnan(dist):
			score1.append(dist)
		else:
			score1.append(0.0)
	score2=[]
	for ii,elem in enumerate(title2):
		#print np.linalg.norm(elem-cat[ii])
		dist = 1 - spatial.distance.cosine(elem, cat2[ii])
		
		if not isnan(dist):
			score2.append(dist)
		else:
			score2.append(0.0)
	score3=[]
	for ii,elem in enumerate(title3):
		#print np.linalg.norm(elem-cat[ii])
		dist = 1 - spatial.distance.cosine(elem, cat3[ii])
		
		if not isnan(dist):
			score3.append(dist)
		else:
			score3.append(0.0)
	score4=[]
	for ii,elem in enumerate(title4):
		#print np.linalg.norm(elem-cat[ii])
		dist = 1 - spatial.distance.cosine(elem, cat4[ii])
		
		if not isnan(dist):
			score4.append(dist)
		else:
			score4.append(0.0)
	
	df2 = pd.DataFrame({'A':score1,'B':score2,'C':score3,'D':score4})
	df2.to_csv(OUT_PATH, index=False, header=False)
	

def tup2str(tpl):

	strn = ''
	for i in tpl:
		strn = strn + i + ' '
	return  strn[:-1]

vocab2vec=cpkl.load(open('Vocab2Vec.p','r'))


data2use_train = map(list,zip(*list(csv.reader(open(TRAIN_DATA_PATH,'r')))))
data2use_test = map(list,zip(*list(csv.reader(open(TEST_DATA_PATH,'r')))))

title_train		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_TITLE_LEN) for line in data2use_train[2]])
cat_train1		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_train[3]])
cat_train2		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_train[4]])
cat_train3		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_train[5]])
cat_train4		=	np.asarray([get_text2wvecs(tup2str(tpl),exclude,vocab2vec,MAX_CAT_LEN) for tpl in zip(data2use_train[3],data2use_train[4],data2use_train[5])])
cat_train = [cat_train1,cat_train2,cat_train3,cat_train4]

title_test		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_TITLE_LEN) for line in data2use_test[2]])
cat_test1		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_test[3]])
cat_test2		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_test[4]])
cat_test3		=	np.asarray([get_text2wvecs(line,exclude,vocab2vec,MAX_CAT_LEN) for line in data2use_test[5]])
cat_test4		=	np.asarray([get_text2wvecs(tup2str(tpl),exclude,vocab2vec,MAX_CAT_LEN) for tpl in zip(data2use_test[3],data2use_test[4],data2use_test[5])])
cat_test = [cat_test1,cat_test2,cat_test3,cat_test4]


T=Input(shape=(300,MAX_TITLE_LEN,), dtype='float32') #300xT
C=Input(shape=(300,MAX_CAT_LEN,), dtype='float32') #300xC
T_T = Lambda(get_transpose, output_shape=transpose_shape)(T)   #Tx300
C_T = Lambda(get_transpose, output_shape=transpose_shape)(C)   #Cx300
C_TU = Dense(300,use_bias=False,kernel_initializer='Identity')(C_T)  #Cx300
G_1 =  Lambda(get_matmul, output_shape=matmul_shape)([C_TU, T]) #CxT
G = Activation(K.tanh)(G_1) #CxT
G_T =  Lambda(get_transpose, output_shape=transpose_shape)(G_1) #TxC
T_maxpool = MaxPooling1D(pool_size=MAX_CAT_LEN, strides=None, padding='valid')(G_1) #1xT
C_maxpool = MaxPooling1D(pool_size=MAX_TITLE_LEN, strides=None, padding='valid')(G_T) #1xC
T_soft = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(T_maxpool)) #Tx1
C_soft = Lambda(get_transpose, output_shape=transpose_shape)(Activation('softmax')(C_maxpool)) #Cx1
outt = Lambda(get_matmul, output_shape=matmul_shape)([T, T_soft]) #300x1
outt_T = Lambda(get_transpose, output_shape=transpose_shape)(outt) #1x300
outc = Lambda(get_matmul, output_shape=matmul_shape)([C, C_soft]) #300x1
outc_T = Lambda(get_transpose, output_shape=transpose_shape)(outc) #1x300
FINAL_EMB = Dense(300,use_bias=False,kernel_initializer='Identity')
outtv = FINAL_EMB(outt_T) #1x300
outcv = FINAL_EMB(outc_T) #1x300
out1 = Reshape((300,))(outtv)
out2 = Reshape((300,))(outcv)
out_final = Dot(axes=1)([out1, out2])
model = Model(inputs=[T, C], outputs=out_final)
emb_title = Model(inputs=[T, C], outputs=out1)
emb_cat = Model(inputs=[T, C], outputs=out2)	


opt = keras.optimizers.Adam(lr=0.001, clipvalue=8.0, clipnorm=5.0)


model.compile(optimizer=opt, loss=losses.categorical_crossentropy)

get4feat(emb_title,emb_cat,title_train,cat_train,OUT_PATH_TR)
get4feat(emb_title,emb_cat,title_test, cat_test,OUT_PATH_TE)
