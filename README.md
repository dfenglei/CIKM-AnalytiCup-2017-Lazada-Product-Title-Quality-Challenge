This ReadMe is divided into four sections:
1) Features Engineering: Run the codes to generate 45 features
2) Shallow Models: Run the shallow approach on test set for both clarity and concise
3) Deep Models: Run the deep approach on test set for both clarity and concise
4) Ensemble: Run the ensemble of the output of two models



Requirements:
			* Keras >= 2.0.2
			* Tensorflow >= 1.0.1 or Theano >= 0.9.0 (We recommend Tensorflow)
			* NLTK >= 3.2.1
			* CUDA-8.0 or CUDNN-5.0
			* Python 2.7 and above
			* LightGbm '2.0.1'		


1) Feature Engineering:

In folder /Submission-CIKM/Features/

There are three folders namely: 
	Input: containing all input files required
* cat1, cat2, cat3.csv contains the all possible states of three categories.
* CCmodel-dict-300d-Small.npy: google glove Common crawl (300) dictionary of all words in train and test set. 

	OutPut: contain all output files
	Scripts: contain all scripts 
We divide feature generation into 6 scripts
features-1: script to generate features from 1-21 and 44-45 (refer to paper)
features-2: script to generate features from 22-27
features-3: script to generate features from 28-30
features-4: script to generate features from 37-39
features-5: script to generate features from 31-36
features-6: script to generate features from 40-43

go to the folder /Submission-CIKM/Features/Scripts/ and run python features-*.py, it will generate features in /Submission-CIKM/Features/Scripts/OutPut folder by the name of Train_Data-features-*.csv and Test_Data-features-*.csv

2) Shallow Models: 

Clarity: 
In folder /Submission-CIKM/Clarity/Shallow/ there are three folders Input, OutPut, Script. Input folder contains the training file with the features used for clarity (as mentioned in paper)

go to /Submission-CIKM/Clarity/Shallow/Scripts/ run python shallow-clarity.py. It will train on training data and will generate an output probabilities on test set (in /Submission-CIKM/Clarity/Shallow/OutPut) by the name "clarity_test.predict"

Concise:
Similarly for conciseness, you can run the code (shallow-concise.py) for concise in the folder /Submission-CIKM/Concise/Shallow/Scripts/

First you need to run the deep and shallow models and get the output for both. After this, run the ensemble to get the final output.


3) Deep Model:
	
Clarity:
cd /Submission-CIKM/Clarity/Deep/
Data to use is in /Submission-CIKM/Clarity/Deep/data/
/Submission-CIKM/Clarity/Deep/Vocab2Vec/ contains 'Vocab2Vec.p' which is the dictionary containing the vocabulary and its corresponding word2vec representation.
/Submission-CIKM/Clarity/Deep/Weights/ contains the saved weights. 

Usage:
	Training : python model_clarity.py --train --learningrate 0.0001 --batchsize 128 
		   tensorboard --logdir /logs		(For viewing progress on tensorboard)
		   

	Validation : python model_clarity.py --validate --weightspath 'weights/weights.hdf5' (For getting the Holdout set results)
		     python model_clarity.py --validate --weightspath 'weights/best.hdf5' (For reproducing our results on Holdout set)
		     

	Testing : python model_clarity.py --test --weightspath 'weights/weights.hdf5' (generates output for test set in the file 'clarity_test.predict')
		  python model_clarity.py --test --weightspath 'weights/best.hdf5' (For reproducing our results on test set)
		  

Conciseness:
cd /Submission-CIKM/Concise/Deep/

Matrices:
	The /Submission-CIKM/Concise/Deep/matrices/ folder stores the precomputed 'Intra-Title Features' matrices (see report)

Usage:
	Training : python model_conciseness.py --train --mats --learningrate 0.0001 --batchsize 128
		   (The --mats argument will compute the 'Intra-Title Features' matrices, you can remove it after the first run as they will be saved in \matrices folder)
		   tensorboard --logdir /logs		(For viewing progress on tensorboard)


	Validation : python model_conciseness.py --validate --weightspath 'weights/weights.hdf5' (For getting Holdout set result)
		     python model_conciseness.py --validate --weightspath 'weights/best.hdf5' (For reproducing our results on Holdout set)


	Testing : python model_conciseness.py --test --weightspath 'weights/weights.hdf5' (generates output for test set in the file 'conciseness_test.predict')
		  python model_conciseness.py --test --weightspath 'weights/best.hdf5' (For reproducing our results on test set)


4) Ensemble:

/Submission-CIKM/Clarity (or Concise)/Ensemble/Input/ contains the output probablities of deep and shallow models on test set.

For Clarity and Conciseness both:
/Submission-CIKM/Clarity/Ensemble/Script/ or /Submission-CIKMClarity/Ensemble/Script/ 
Usage:
	python ensemble.py
Output:
	The output will be stored in ../Output/clarity_test.predict or ../Output/conciseness_test.predict 
# CIKM-AnalytiCup-2017-Lazada-Product-Title-Quality-Challenge
