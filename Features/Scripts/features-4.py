import pandas as pd
import numpy as np
import re
import scipy.spatial.distance as dis
import collections

model = np.load("./../Input/CCmodel-dict-300d-Small.npy").item()

train = pd.read_csv("./../Input/data_train.csv")
test = pd.read_csv("./../Input/data_test.csv")
#train_shape = train.shape
#file = train
file = train.append(test, ignore_index=True)

title = file.Title
cat1 = file.Cat1
cat2 = file.Cat2
cat3 = file.Cat3

list_matrixAvgCat1 = []
list_matrixMaxCat1= []
list_matrixMinCat1= []
list_matrixSumCat1= []

list_matrixAvgCat2 = []
list_matrixMaxCat2= []
list_matrixMinCat2= []
list_matrixSumCat2= []

list_matrixAvgCat3 = []
list_matrixMaxCat3= []
list_matrixMinCat3= []
list_matrixSumCat3= []


for k in range(0, len(title)):
	print k
	count=0
	count_total = 0
	
	clean_title = re.sub('[^a-zA-Z0-9\n]', ' ', title[k])
	clean_title = re.sub( '\s+', ' ', clean_title).strip()
		                         
	splitTitle = str(clean_title).split(" ")
	splitTitle = [x.lower() for x in splitTitle]
	#print splitTitle1
	#splitTitle = [value for value in splitTitle1 if value!='nbsp' and value!='amp']
	countTitle = 0.0
	#vecTitle = np.zeros(size)
	#print splitTitle
	ListTitle = []
	for word in splitTitle:
		if(model.has_key(word)):
			ListTitle.append(word)
	#print "Title is %s" %ListTitle
	################Cat1###########################################
	
	if((str(cat1[k])!= 'nan' and len(ListTitle)!= 0)):
		clean_cat1 = re.sub('[^a-zA-Z0-9\n]', ' ', cat1[k])
		clean_cat1 = re.sub( '\s+', ' ', clean_cat1).strip()
		splitCat1 = str(clean_cat1).split(" ")
		splitCat1 = [x.lower() for x in splitCat1]
		ListCat1 = []
		for word in splitCat1:
			if(model.has_key(word)):
				ListCat1.append(word)		
		if(len(ListCat1)!=0):
			Matrix = np.zeros((len(ListTitle),len(ListCat1)),dtype=np.float)
			count =0
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat1)):
						Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[ListCat1[j]])
			List_matrix = []
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat1)):
						List_matrix.append(Matrix[i,j])
			arr_matrixCat1 = np.asarray(List_matrix)
			arr_matrixCat1 = (arr_matrixCat1 + 1)/2
			max_countCat1 = arr_matrixCat1.max()
			sum_countCat1 = arr_matrixCat1.sum()
			min_countCat1 = arr_matrixCat1.min()
			#min_count = sum(1 for k in arr_matrix if k<=0.1)
			#full_matchCount = sum(1 for k in arr_matrix if k==1)
			avg_matrixCat1 = arr_matrixCat1.mean()
			#print " list of cat1 is %s" %ListCat1			
			list_matrixAvgCat1.append(avg_matrixCat1)
			list_matrixMaxCat1.append(max_countCat1)
			list_matrixMinCat1.append(min_countCat1)
			list_matrixSumCat1.append(sum_countCat1)
		else:
			list_matrixAvgCat1.append(-1)
			list_matrixMaxCat1.append(-1)
			list_matrixMinCat1.append(-1)
			list_matrixSumCat1.append(-1)			
	else:
		list_matrixAvgCat1.append(-1)
		list_matrixMaxCat1.append(-1)
		list_matrixMinCat1.append(-1)
		list_matrixSumCat1.append(-1)		
	
	################Cat2###########################################
	#print cat2[k]
	if((str(cat2[k])!= 'nan' and len(ListTitle)!= 0)):
		clean_cat2 = re.sub('[^a-zA-Z0-9\n]', ' ', cat2[k])
		clean_cat2 = re.sub( '\s+', ' ', clean_cat2).strip()
		splitCat2 = str(clean_cat2).split(" ")
		splitCat2 = [x.lower() for x in splitCat2]
		ListCat2 = []
		for word in splitCat2:
			if(model.has_key(word)):
				ListCat2.append(word)		
		if(len(ListCat2)!=0):
			Matrix = np.zeros((len(ListTitle),len(ListCat2)),dtype=np.float)
			count =0
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat2)):
						Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[ListCat2[j]])
			List_matrix = []
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat2)):
						List_matrix.append(Matrix[i,j])
			arr_matrixCat2 = np.asarray(List_matrix)
			arr_matrixCat2 = (arr_matrixCat2 + 1)/2
			#max_count = sum(1 for k in arr_matrix if k>=0.85)
			#min_count = sum(1 for k in arr_matrix if k<=0.1)
			#full_matchCount = sum(1 for k in arr_matrix if k==1)
			avg_matrixCat2 = arr_matrixCat2.mean()
			max_countCat2 = arr_matrixCat2.max()
			sum_countCat2 = arr_matrixCat2.sum()
			min_countCat2 = arr_matrixCat2.min()
			#print " list of cat1 is %s" %ListCat1			
			list_matrixAvgCat2.append(avg_matrixCat2)
			list_matrixMaxCat2.append(max_countCat2)
			list_matrixMinCat2.append(min_countCat2)
			list_matrixSumCat2.append(sum_countCat2)
		else:
			list_matrixAvgCat2.append(-1)
			list_matrixMaxCat2.append(-1)
			list_matrixMinCat2.append(-1)
			list_matrixSumCat2.append(-1)			
	else:
		list_matrixAvgCat2.append(-1)
		list_matrixMaxCat2.append(-1)
		list_matrixMinCat2.append(-1)
		list_matrixSumCat2.append(-1)		
	################Cat3###########################################
	
	if((str(cat3[k])!= 'nan' and len(ListTitle)!= 0)):
		clean_cat3 = re.sub('[^a-zA-Z0-9\n]', ' ', cat3[k])
		clean_cat3 = re.sub( '\s+', ' ', clean_cat3).strip()
		splitCat3 = str(clean_cat3).split(" ")
		splitCat3 = [x.lower() for x in splitCat3]
		ListCat3 = []
		for word in splitCat3:
			if(model.has_key(word)):
				ListCat3.append(word)					
		if(len(ListCat3)!=0):
			Matrix = np.zeros((len(ListTitle),len(ListCat3)),dtype=np.float)
			count =0
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat3)):
						Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[ListCat3[j]])
			List_matrix = []
			for i in range(0,len(ListTitle)):
				for j in range(0,len(ListCat3)):
						List_matrix.append(Matrix[i,j])
			arr_matrixCat3 = np.asarray(List_matrix)
			arr_matrixCat3 = (arr_matrixCat3 + 1)/2
			#max_count = sum(1 for k in arr_matrix if k>=0.85)
			#min_count = sum(1 for k in arr_matrix if k<=0.1)
			#full_matchCount = sum(1 for k in arr_matrix if k==1)
			avg_matrixCat3 = arr_matrixCat3.mean()
			max_countCat3 = arr_matrixCat3.max()
			sum_countCat3 = arr_matrixCat3.sum()
			min_countCat3 = arr_matrixCat3.min()
			#print " list of cat1 is %s" %ListCat1			
			list_matrixAvgCat3.append(avg_matrixCat3)
			list_matrixMaxCat3.append(max_countCat3)
			list_matrixMinCat3.append(min_countCat3)
			list_matrixSumCat3.append(sum_countCat3)
		else:
			list_matrixAvgCat3.append(-1)
			list_matrixMaxCat3.append(-1)
			list_matrixMinCat3.append(-1)
			list_matrixSumCat3.append(-1)			
	else:
		list_matrixAvgCat3.append(-1)
		list_matrixMaxCat3.append(-1)
		list_matrixMinCat3.append(-1)
		list_matrixSumCat3.append(-1)	
	

#file['AvgOfSemDistanceCat1'] = list_matrixAvgCat1
#file['AvgOfSemDistanceCat2'] = list_matrixAvgCat2
#file['AvgOfSemDistanceCat3'] = list_matrixAvgCat3

#file['SumOfSemDistanceCat1'] = list_matrixSumCat1
#file['SumOfSemDistanceCat2'] = list_matrixSumCat2
#file['SumOfSemDistanceCat3'] = list_matrixSumCat3

file['MaxOfSemDistanceCat1'] = list_matrixMaxCat1
file['MaxOfSemDistanceCat2'] = list_matrixMaxCat2
file['MaxOfSemDistanceCat3'] = list_matrixMaxCat3

#file['MinOfSemDistanceCat1'] = list_matrixMinCat1
#file['MinOfSemDistanceCat2'] = list_matrixMinCat2
#file['MinOfSemDistanceCat3'] = list_matrixMinCat3



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

#train_new.to_csv("./Train_Data.csv", index = False)
#test_new.to_csv("./Test_Data.csv", index = False)

train_new.to_csv("./../OutPut/Train_Data-features-4.csv", index = False)
test_new.to_csv("./../OutPut/Test_Data-features-4.csv", index = False)

