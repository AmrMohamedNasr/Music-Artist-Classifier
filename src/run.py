from utils import *
from model import *
import numpy as np
import os

def run(conf, path):
	x = prediction_data_pipeline(conf, path)
	model = build_model(conf)
	mod_x = []
	for feat in x:
		mod_x.append(np.stack(feat))
	y = predict_model(conf, model, mod_x)
	y = np.argmax(y, axis = 1)
	votes = np.bincount(y)
	unique_composers = []
	print('Votes for artisits : ')
	for filename in os.listdir(conf['dataset']['raw_path']):
		unique_composers.append(filename)
	for i in range(len(unique_composers)):
		if (i < len(votes)):
			print('\t' + unique_composers[i] + ' : ', votes[i])
		else:
			print('\t' + unique_composers[i] + ' : ', 0)
	prediction = np.argmax(votes)
	print('Prediction is : ', unique_composers[prediction])
