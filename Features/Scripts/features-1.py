import pandas as pd
import numpy as np
import re
from pyjarowinkler import distance
import collections
train = pd.read_csv("./../Input/data_train.csv")
test = pd.read_csv("./../Input/data_test.csv")

file = train.append(test, ignore_index=True)
#file = train

file.Price = file.Price.replace(-1,0)
#LogTransform due to skewness
file.Price = np.log1p(file.Price)
#file.to_csv("./sample.csv")
#Creating new columns with number of words in product title and if it contain digit
title = file.Title
print title.shape
list_size = []
list_sizeSpace = []
list_bool=[]
list_nonAlpha = []
list_maxLen = []
list_maxLenSpace = []
list_minLen = []
list_minLenSpace = []
list_avgLen = []
list_avgLenSpace = []
list_matrixSum = []
list_matrixAvg = []
list_matrixMaxCount = []
list_matrixMinCount = []
list_matrixFullMatchCount = []
ListOfSexy = []
list_caps = []

for i in range(len(title)):
	#print title[i]
	print i
	count=0
	count_total = 0
	
	bool_digit = any(j.isdigit() for j in title[i])			#Check if title contains digits
	list_bool.append(bool_digit)
	
	for char in title[i]:	
		if(char!= " "):
			count_total += 1
		if not char.isalpha():
			if(char!= " "):
				count += 1
	#print count
	#print count_total
	caps = 0
	caps = sum(1 for c in title[i] if c.isupper())
				
	per_count = float(count)/float(count_total)       	  #Percentage of non-alphabatic characters
	list_nonAlpha.append(per_count)
	
	CapsperCount = float(caps)/float(count_total)
	list_caps.append(CapsperCount)
	
	clean_title = re.sub('[^a-zA-Z0-9\n\.]', ' ', title[i])    #clean title
	clean_title = re.sub( '\s+', ' ', clean_title).strip()
	#print clean_title
	size = len(str(clean_title).split(" "))
	sizeSpace = len(str(title[i]).split(" "))
	#print size
	list_size.append(size)
	list_sizeSpace.append(sizeSpace)		                         #Count number of words		
	
	splitTitle = str(clean_title).split(" ")
	splitTitle = [x.lower() for x in splitTitle]
	boolDigit =  any(s for s in splitTitle if (s == 'sexy'))
	ListOfSexy.append(boolDigit)
	
	#print splitTitle1
	#splitTitle = [value for value in splitTitle1 if value!='nbsp' and value!='amp']
	#print splitTitle
	NormTitle = str(title[i]).split(" ")
	#print "
	list_words = []
	for k in range(0, len(splitTitle)):
		#print splitTitle[k]
		list_words.append(len(splitTitle[k]))
	max_len = max(list_words)
	min_len = min(list_words)
	avg_len = float(sum(list_words))/float(size)
	#print len(list_words)
	list_avgLen.append(avg_len)                       #average len of title
	#print "hi"
	list_maxLen.append(max_len)                       #max length word in title
	list_minLen.append(min_len)							#min length of word
	
	list_wordsSpace = []
	for k in range(0, len(NormTitle)):
		list_wordsSpace.append(len(NormTitle[k]))
	max_lenSpace = max(list_wordsSpace)
	min_lenSpace = min(list_wordsSpace)
	avg_lenSpace = float(sum(list_wordsSpace))/float(sizeSpace)
	#print len(list_words)
	list_avgLenSpace.append(avg_lenSpace)
	#print "hi"
	list_maxLenSpace.append(max_lenSpace)
	list_minLenSpace.append(min_lenSpace)
	
	
	List1 = splitTitle
	List2 = splitTitle
	Matrix = np.zeros((len(List1),len(List2)),dtype=np.float)
	count =0
	for i in range(0,len(List1)):
		for j in range(0,len(List2)):
			Matrix[i,j] = distance.get_jaro_distance(List1[i],List2[j])
			#if(Matrix[i,j]==1 and List1[i] == 'nbsp'):
			#	count+= 1
			#print "for %s and %s distance is %f" %( List1[i], List2[j], Matrix[i,j])
	List_matrix = []
	for i in range(0,len(List1)):
		for j in range(0,len(List2)):
			if(i!=j):
				List_matrix.append(Matrix[i,j])
	arr_matrix = np.asarray(List_matrix)
	max_count = sum(1 for k in arr_matrix if k>=0.85)
	min_count = sum(1 for k in arr_matrix if k<=0.1)
	full_matchCount = sum(1 for k in arr_matrix if k==1)
	
	
	avg_matrix = arr_matrix.mean()				
	#print avg_matrix		 
	#Matrix = Matrix.reshape(Matrix.shape[0]*Matrix.shape[1])
	#print Matrix.size			
	#list_matrixSum.append(Matrix.sum())
		
	size_arr = arr_matrix.size
	sum_matrix = arr_matrix.sum()
	#counts = collections.Counter(Matrix)
	#max_count = counts.get(Matrix.max())
	#min_count = counts.get(Matrix.min())
	if(size_arr!=0):
			list_matrixMaxCount.append(float(max_count)/float(arr_matrix.size))
			list_matrixMinCount.append(float(min_count)/float(arr_matrix.size))	
			list_matrixFullMatchCount.append(float(full_matchCount)/float(arr_matrix.size))
			list_matrixAvg.append(avg_matrix)
			list_matrixSum.append(sum_matrix)	
	else:
			list_matrixMaxCount.append(0)
			list_matrixMinCount.append(0)	
			list_matrixFullMatchCount.append(0)
			list_matrixAvg.append(0)
			list_matrixSum.append(0)		
	
file['NumberOfWordsTitle'] = list_size
file['NumberOfWordsTitleSpace'] = list_sizeSpace        
file['ContainsDigit'] = list_bool 
file['ContainsDigit'] = file['ContainsDigit'].astype(int)
file['NoOfNonAlpha'] = list_nonAlpha
file['AvgLength'] = list_avgLen
file['MaxLength'] = list_maxLen
file['MinLength'] = list_minLen
file['AvgLengthSpace'] = list_avgLenSpace
file['MaxLengthSpace'] = list_maxLenSpace
file['MinLengthSpace'] = list_minLenSpace
file['SumOfStringDistance'] = list_matrixSum
file['AvgOfStringDistance'] = list_matrixAvg
file['MaxStringDistCount'] = list_matrixMaxCount
file['MinStringDistCount'] = list_matrixMinCount
file['FullMatchCount'] = list_matrixFullMatchCount
file['sexy'] = ListOfSexy 
file['sexy'] = file['sexy'].astype(int)	
file['perOfCaps'] = list_caps


#file.to_csv("./Train-Data-1.csv")

#Convert categorical features into numerical
file.Cat1 = file.Cat1.astype('category')
file.Cat2 = file.Cat2.astype('category')
file.Cat3 = file.Cat3.astype('category')
file.Country = file.Country.astype('category')
file.level = file.level.astype('category')
cat_columns = file.select_dtypes(['category']).columns
#df1 = pd.get_dummies(file[cat_columns])
#file.drop(file[cat_columns], axis =1, inplace=True)
#file = file.join(df1)
file[cat_columns] = file[cat_columns].apply(lambda x: x.cat.codes)


#Creating a new dataframe of chosen columns
#df = file[['Country', 'Cat1', 'Cat2', 'Cat3', 'Price', 'level', 'NumberOfWordsTitle', 'ContainsDigit', 'NoOfNonAlpha']].copy()
file.drop('Product', axis=1, inplace=True)
file.drop('Title', axis=1, inplace=True)
file.drop('Prod-detail', axis=1, inplace=True)

df = file
train_new = df[:train.shape[0]]
test_new = df[train.shape[0]:]
clarity = pd.read_csv("./../Input/clarity_train-labels.txt", header=None)
concise = pd.read_csv("./../Input/conciseness_train-labels.txt", header=None)
train_new['clarity'] = clarity
train_new['concise'] = concise


train_new.to_csv("./../OutPut/Train_Data-features-1.csv", index = False)
test_new.to_csv("./../OutPut/Test_Data-features-1.csv", index = False)

