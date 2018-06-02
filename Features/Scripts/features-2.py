import numpy as np
import pandas as pd
import re
import scipy.spatial.distance as dis

model = np.load("./../Input/CCmodel-dict-300d-Small.npy").item()
#model = np.load("./model-dict50d.npy").item()
train = pd.read_csv("./../Input/data_train.csv")
test = pd.read_csv("./../Input/data_test.csv")
file = train.append(test, ignore_index=True)
title = file.Title
list_matrixAvg = []
list_matrixMaxCount = []
list_matrixMinCount = []
list_matrixFullMatchCount = []
list_PerPresentInTitle = []
list_UniqueNonPresent = []
print "loaded"
print len(model)
for i in range(0, len(title)):
	print i		
	clean_title = re.sub('[^a-zA-Z0-9\n\.]', ' ', title[i])
	clean_title = re.sub( '\s+', ' ', clean_title).strip()
	#print clean_title
	size = len(str(clean_title).split(" "))
	#print size
	#list_size.append(size)		                         #Count number of words		
	
	splitTitle = str(clean_title).split(" ")
	splitTitle = [x.lower() for x in splitTitle]
	#print splitTitle
	List1 = []
	notInDict= set()
	for word in splitTitle:
		if(model.has_key(word)):
			List1.append(word)
		else:
			#print word
			notInDict.add(word)		
	List2 = List1
	presntSize = len(List1)
	#print List1
	setSize = len(notInDict)
	
	non_presenters = size - presntSize
	if(non_presenters!=0):
		list_UniqueNonPresent.append(float(setSize)/float(non_presenters))
		#list_UniqueNonPresent.append(setSize)
	else:
		list_UniqueNonPresent.append(0)	
	#print non_presenters
	#print len(splitTitle)
	#print presntSize
	if(size!=0):
		list_PerPresentInTitle.append(float(presntSize)/float(size))
	else:
		list_PerPresentInTitle.append(0)
	#list_matrixPresentPer.append(sizeOfPresenters)		
	Matrix = np.zeros((presntSize,presntSize),dtype=np.float)
	
	for i in range(0,presntSize):
		for j in range(0,presntSize):
			Matrix[i,j] = 1 - dis.cosine(model[List1[i]],model[List2[j]])
			#print List1[i], List2[j], Matrix[i,j]
	List_matrix = []
	for i in range(0,len(List1)):
		for j in range(0,len(List2)):
			if(i!=j):
				List_matrix.append(Matrix[i,j])
	arr_matrix = np.asarray(List_matrix)
	arr_matrix = (arr_matrix+1)/2
	max_count = sum(1 for k in arr_matrix if k>=0.85)
	min_count = sum(1 for k in arr_matrix if k<=0.5)
	full_matchCount = sum(1 for k in arr_matrix if k>=0.95)
		
	size_arr = arr_matrix.size
	if(size_arr!=0):
			list_matrixMaxCount.append(float(max_count)/float(arr_matrix.size))
			list_matrixMinCount.append(float(min_count)/float(arr_matrix.size))	
			list_matrixFullMatchCount.append(float(full_matchCount)/float(arr_matrix.size))
			avg_matrix = arr_matrix.mean()
			list_matrixAvg.append(avg_matrix)	
	else:
			list_matrixMaxCount.append(0)
			list_matrixMinCount.append(0)	
			list_matrixFullMatchCount.append(0)
			list_matrixAvg.append(0)		

file['AvgOfSemanticDistance'] = list_matrixAvg
file['MaxSemanticDistCount'] = list_matrixMaxCount
file['MinSemanticDistCount'] = list_matrixMinCount
file['FullSemanticCount'] = list_matrixFullMatchCount
file['PerPresentInTitle'] = list_PerPresentInTitle
file['UniqueNonPresenter'] = list_UniqueNonPresent

file.drop('Product', axis=1, inplace=True)
file.drop('Title', axis=1, inplace=True)
file.drop('Prod-detail', axis=1, inplace=True)
file.drop('Cat1', axis=1, inplace=True)
file.drop('Cat2', axis=1, inplace=True)
file.drop('Cat3', axis=1, inplace=True)
file.drop('level', axis=1, inplace=True)
file.drop('Price', axis=1, inplace=True)
file.drop('Country', axis=1, inplace=True)

df = file
train_new = df[:train.shape[0]]
test_new = df[train.shape[0]:]
#clarity = pd.read_csv("./clarity_train-labels.txt", header=None)
#concise = pd.read_csv("./conciseness_train-labels.txt", header=None)
#train_new['clarity'] = clarity
#train_new['concise'] = concise

train_new.to_csv("./../OutPut/Train_Data-features-2.csv", index = False)
test_new.to_csv("./../OutPut/Test_Data-features-2.csv", index = False)
