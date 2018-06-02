#!/usr/bin/env python2
import csv
import argparse

OUTPUT_CON = './../Output/conciseness_test.predict'
TestList_con = map(list,zip(*list(csv.reader(open('./../Input/OutPutOfDeepandShallow-Con-Test.csv','r')))))
WeightList_con = [0.5,0.5]


def getOutList(TestList,WeightList,OUT_FILE):
	OutputList = []
	for i,elem in enumerate(TestList[0]):
		if i == 0:
			continue
		if elem > 2:
			OutProb = WeightList[0]*float(TestList[1][i]) + WeightList[1]*float(TestList[2][i])
			OutputList.append(OutProb)
		else:
			OutputList.append(float(TestList[1][i]))

	fw = open(OUT_FILE,'w')
	for elem in OutputList:
		fw.write(str(elem)+'\n')
	fw.close()
	

getOutList(TestList_con,WeightList_con,OUTPUT_CON)

