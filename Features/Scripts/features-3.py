import numpy as np
import pandas as pd
import re
import scipy.spatial.distance as dis

model = np.load("./../Input/CCmodel-dict-300d-Small.npy").item()

train = pd.read_csv("./../Input/data_train.csv")
test = pd.read_csv("./../Input/data_test.csv")
file = train.append(test, ignore_index=True)
title = file.Title
cat1 = file.Cat1
cat2 = file.Cat2
cat3 = file.Cat3
size = 300
list_disCat1 = []
list_disCat2 = []
list_disCat3 = []


print "loaded"
for i in range(0, len(title)):
	print i	
	clean_title = re.sub('[^a-zA-Z0-9\n\.]', ' ', title[i])
	clean_title = re.sub( '\s+', ' ', clean_title).strip()
	splitTitle = str(clean_title).split(" ")
	splitTitle = [x.lower() for x in splitTitle]
	
	countTitle = 0.0
	vecTitle = np.zeros(size)
	#print splitTitle
	for word in splitTitle:
		if(model.has_key(word)):
			#print word
			countTitle+=1
			vecTitle+= np.array(model[word])
	avgVecTitle = vecTitle/countTitle
	
############################Cat1#########################
		
	if(str(cat1[i])!= 'nan') and (countTitle != 0.0):
		clean_cat1 = re.sub('[^a-zA-Z0-9\n\.]', ' ', cat1[i])
		clean_cat1 = re.sub( '\s+', ' ', clean_cat1).strip()
		splitCat1 = str(clean_cat1).split(" ")
		splitCat1 = [x.lower() for x in splitCat1]
		
		countCat1 = 0.0
		vecCat1 = np.zeros(size)
		for word in splitCat1:
			if(model.has_key(word)):
				countCat1+=1
				vecCat1+= np.array(model[word])
		avgVecCat1 = vecCat1/countCat1
		avgDisTitleCat1 = 1 - dis.cosine(avgVecTitle, avgVecCat1)
		avgDisTitleCat1 = (avgDisTitleCat1+1)/2
		list_disCat1.append(avgDisTitleCat1)
	else:
		list_disCat1.append(-1)	
#######################Cat2#################################
	if(str(cat2[i])!= 'nan') and (countTitle != 0.0):
		clean_cat2 = re.sub('[^a-zA-Z0-9\n\.]', ' ', cat2[i])
		clean_cat2 = re.sub( '\s+', ' ', clean_cat2).strip()
		splitCat2 = str(clean_cat2).split(" ")
		splitCat2 = [x.lower() for x in splitCat2]
		
		countCat2 = 0.0
		vecCat2 = np.zeros(size)
		for word in splitCat2:
			if(model.has_key(word)):
				countCat2+=1
				vecCat2+= np.array(model[word])
		avgVecCat2 = vecCat2/countCat2
		avgDisTitleCat2 = 1 - dis.cosine(avgVecTitle, avgVecCat2)
		avgDisTitleCat2 = (avgDisTitleCat2+1)/2
		list_disCat2.append(avgDisTitleCat2)	
	else:
		list_disCat2.append(-1)	
	
#####################Cat3##########################################	
	if(str(cat3[i])!= 'nan') and (countTitle != 0.0):
		clean_cat3 = re.sub('[^a-zA-Z0-9\n\.]', ' ', cat3[i])
		clean_cat3 = re.sub( '\s+', ' ', clean_cat3).strip()
		splitCat3 = str(clean_cat3).split(" ")
		splitCat3 = [x.lower() for x in splitCat3]
		
		countCat3 = 0.0
		vecCat3 = np.zeros(size)
		for word in splitCat3:
			if(model.has_key(word)):
				countCat3+=1
				vecCat3+= np.array(model[word])
		if (countCat3 ==0):
				list_disCat3.append(-1)
		else:
			avgVecCat3 = vecCat3/countCat3	
			avgDisTitleCat3 = 1 - dis.cosine(avgVecTitle, avgVecCat3)
			avgDisTitleCat3 = (avgDisTitleCat3+1)/2	
			list_disCat3.append(avgDisTitleCat3)				
	else:
		list_disCat3.append(-1)	
	
file['AvgDisTitleCat1'] = list_disCat1
file['AvgDisTitleCat2'] = list_disCat2
file['AvgDisTitleCat3'] = list_disCat3

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

train_new.to_csv("./../OutPut/Train_Data-features-3.csv", index = False)
test_new.to_csv("./../OutPut/Test_Data-features-3.csv", index = False)
