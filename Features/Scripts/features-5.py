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
cat1own = file.Cat1
cat2own = file.Cat2
cat1File = pd.read_csv("./../Input/cat1.csv")
cat2File = pd.read_csv("./../Input/cat2.csv")
#cat3File = pd.read_csv("./cat3.csv")

cat1 = cat1File.Cat1
cat2 = cat2File.Cat2
#cat3 = cat3File.Cat3

global_Maxcat2 = []
global_Mincat2 = []
globalSumcat2 = []
global_Maxcat1 = []
global_Mincat1 = []
globalSumcat1 = []

#global_Avgcat3 = []
#globalSumcat3 = []

for k in range(0, len(title)):
	list_matrixAvgcat1 = []
	#list_matrixSumcat1 = []
	list_matrixAvgcat2 = []
	list_matrixMaxcat1 = []
	list_matrixMincat1 = []
	#list_matrixSumcat2 = []
	list_matrixMaxcat2 = []
	list_matrixMincat2 = []
	#list_matrixAvgcat3 = []
	#list_matrixSumcat3 = []
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
	for l in range(0, len(cat1)):
		if((str(cat1[l])!= 'nan' and len(ListTitle)!= 0) and cat1own[k]!=cat1[l]):
			#print cat1[l]
			clean_cat1 = re.sub('[^a-zA-Z0-9\n]', ' ', cat1[l])
			clean_cat1 = re.sub( '\s+', ' ', clean_cat1).strip()
			splitcat1 = str(clean_cat1).split(" ")
			splitcat1 = [x.lower() for x in splitcat1]
			Listcat1 = []
			for word in splitcat1:
				if(model.has_key(word)):
					Listcat1.append(word)		
			if(len(Listcat1)!=0):
				Matrix = np.zeros((len(ListTitle),len(Listcat1)),dtype=np.float)
				count =0
				for i in range(0,len(ListTitle)):
					for j in range(0,len(Listcat1)):
							Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[Listcat1[j]])
				List_matrix = []
				for i in range(0,len(ListTitle)):
					for j in range(0,len(Listcat1)):
							List_matrix.append(Matrix[i,j])
				arr_matrixcat1 = np.asarray(List_matrix)
				arr_matrixcat1 = (arr_matrixcat1 + 1)/2
				#max_count = sum(1 for k in arr_matrix if k>=0.85)
				#min_count = sum(1 for k in arr_matrix if k<=0.1)
				#full_matchCount = sum(1 for k in arr_matrix if k==1)
				avg_matrixcat1 = arr_matrixcat1.mean()
				max_matrixCat1 = arr_matrixcat1.max()
				min_matrixCat1 = arr_matrixcat1.min()
				#print "mean between for %s is %f" %(cat1[l], avg_matrixcat1)
				#print Matrix
				#sum_matrixcat1 = arr_matrixcat1.sum()
				#print " list of cat1 is %s" %Listcat1			
			list_matrixAvgcat1.append(avg_matrixcat1)
			list_matrixMaxcat1.append(max_matrixCat1)
			list_matrixMincat1.append(min_matrixCat1)
			#list_matrixSumcat1.append(sum_matrixcat1)
	if(len(list_matrixAvgcat1)!=0):				
		sumGlobalcat1 = sum(list_matrixAvgcat1)
		maxOfMaxcat1 = max(list_matrixMaxcat1)
		minOfMincat1 = min(list_matrixMincat1)
		#avgGlobalcat1 = float(sum(list_matrixAvgcat1))/float(len(list_matrixAvgcat1))		
		#global_Avgcat1.append(avgGlobalcat1)
		globalSumcat1.append(sumGlobalcat1)
		global_Maxcat1.append(maxOfMaxcat1)
		global_Mincat1.append(minOfMincat1)
	else:
		#global_Avgcat1.append(-1)
		globalSumcat1.append(-1)
		global_Maxcat1.append(-1)
		global_Mincat1.append(-1)									
	################Cat2###########################################
	for l in range(0, len(cat2)):
		if((str(cat2[l])!= 'nan' and len(ListTitle)!= 0) and cat2own[k]!=cat2[l]):
			clean_cat2 = re.sub('[^a-zA-Z0-9\n]', ' ', cat2[l])
			clean_cat2 = re.sub( '\s+', ' ', clean_cat2).strip()
			splitcat2 = str(clean_cat2).split(" ")
			splitcat2 = [x.lower() for x in splitcat2]
			Listcat2 = []
			for word in splitcat2:
				if(model.has_key(word)):
					Listcat2.append(word)		
			if(len(Listcat2)!=0):
				Matrix = np.zeros((len(ListTitle),len(Listcat2)),dtype=np.float)
				count =0
				for i in range(0,len(ListTitle)):
					for j in range(0,len(Listcat2)):
							Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[Listcat2[j]])
				List_matrix = []
				for i in range(0,len(ListTitle)):
					for j in range(0,len(Listcat2)):
							List_matrix.append(Matrix[i,j])
				arr_matrixcat2 = np.asarray(List_matrix)
				arr_matrixcat2 = (arr_matrixcat2 + 1)/2
				#max_count = sum(1 for k in arr_matrix if k>=0.85)
				#min_count = sum(1 for k in arr_matrix if k<=0.1)
				#full_matchCount = sum(1 for k in arr_matrix if k==1)
				avg_matrixcat2 = arr_matrixcat2.mean()
				#sum_matrixcat2 = arr_matrixcat2.sum()
				max_matrixCat2 = arr_matrixcat2.max()
				min_matrixCat2 = arr_matrixcat2.min()
				#print " list of cat2 is %s" %Listcat2			
			list_matrixAvgcat2.append(avg_matrixcat2)
			list_matrixMaxcat2.append(max_matrixCat2)
			list_matrixMincat2.append(min_matrixCat2)
			#list_matrixSumcat2.append(sum_matrixcat2)
	if(len(list_matrixAvgcat2)!=0):		
		sumGlobalcat2 = sum(list_matrixAvgcat2)
		maxOfMaxcat2 = max(list_matrixMaxcat2)
		minOfMincat2 = min(list_matrixMincat2)
		#avgGlobalcat1 = float(sum(list_matrixAvgcat1))/float(len(list_matrixAvgcat1))		
		#global_Avgcat1.append(avgGlobalcat1)
		globalSumcat2.append(sumGlobalcat2)
		global_Maxcat2.append(maxOfMaxcat2)
		global_Mincat2.append(minOfMincat2)
	else:
		globalSumcat2.append(-1)
		global_Maxcat2.append(-1)
		global_Mincat2.append(-1)				
		#################Cat3###########################################
	#for l in range(0, len(cat3)):
		#if((str(cat3[l])!= 'nan' and len(ListTitle)!= 0)):
			#clean_cat3 = re.sub('[^a-zA-Z0-9\n]', ' ', cat3[l])
			#clean_cat3 = re.sub( '\s+', ' ', clean_cat3).strip()
			#splitcat3 = str(clean_cat3).split(" ")
			#splitcat3 = [x.lower() for x in splitcat3]
			#Listcat3 = []
			#for word in splitcat3:
				#if(model.has_key(word)):
					#Listcat3.append(word)		
			#if(len(Listcat3)!=0):
				#Matrix = np.zeros((len(ListTitle),len(Listcat3)),dtype=np.float)
				#count =0
				#for i in range(0,len(ListTitle)):
					#for j in range(0,len(Listcat3)):
							#Matrix[i,j] = 1 - dis.cosine(model[ListTitle[i]],model[Listcat3[j]])
				#List_matrix = []
				#for i in range(0,len(ListTitle)):
					#for j in range(0,len(Listcat3)):
							#List_matrix.append(Matrix[i,j])
				#arr_matrixcat3 = np.asarray(List_matrix)
				#arr_matrixcat3 = (arr_matrixcat3 + 1)/2
				##max_count = sum(1 for k in arr_matrix if k>=0.85)
				##min_count = sum(1 for k in arr_matrix if k<=0.1)
				##full_matchCount = sum(1 for k in arr_matrix if k==1)
				#avg_matrixcat3 = arr_matrixcat3.mean()
				#sum_matrixcat3 = arr_matrixcat3.sum()
				##print " list of cat1 is %s" %ListCat1			
			#list_matrixAvgcat3.append(avg_matrixcat3)
			#list_matrixSumcat3.append(sum_matrixcat3)
	#if(len(list_matrixAvgcat3!=0)):			
		#sumGlobalcat3 = sum(list_matrixSumcat3)
		#avgGlobalcat3 = float(sum(list_matrixAvgcat3))/float(len(list_matrixAvgcat3))		
		#global_Avgcat3.append(avgGlobalcat3)
		#globalSumcat3.append(sumGlobalcat3)
	#else:
		#global_Avgcat3.append(-1)
		#globalSumcat3.append(-1)			
	

file['SumAllTitlesCat2'] = globalSumcat2
file['MaxofTitlesCat2'] = global_Maxcat2
file['MinofTitlesCat2'] = global_Mincat2
file['SumAllTitlesCat1'] = globalSumcat1
file['MaxofTitlesCat1'] = global_Maxcat1
file['MinofTitlesCat1'] = global_Mincat1


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


#train_new.to_csv("./Train_Data.csv", index = False)
#test_new.to_csv("./Test_Data.csv", index = False)

train_new.to_csv("./../OutPut/Train_Data-features-5.csv", index = False)
test_new.to_csv("./../OutPut/Test_Data-features-5.csv", index = False)

