from feature_extraction import *
from data_load import *
from keras.utils import to_categorical
from data_augmentation import *
from sklearn.model_selection import train_test_split
import os
import numpy as np
'''
combine_features function adds features to each pianoroll
returns:
combined_x -- combined pianoroll
'''
def combine_features(piano_roll, feature):
	combined_x = []
	for i in range(len(piano_roll)):
		combined_x.append([])
		combined_x[i].append(piano_roll[i])
		for j in range(len(feature)):
			combined_x[i].append(feature[j][i])
	return combined_x
'''
uncombine_features function divides combined pianoroll into pianoroll and features.
returns:
piano_roll -- pianoroll.
features -- features of pianoroll.
'''
def uncombine_features(combined_x):
	piano_roll = []
	features = []
	feature_num = len(combined_x[0]) - 1
	for i in range(feature_num):
		features.append([])
	for i in range(len(combined_x)):
		piano_roll.append(combined_x[i][0])
		for j in range(feature_num):
			features[j].append(combined_x[i][j + 1])
	return piano_roll, features

'''
full_data_pipeline function reads dataset and preprocess midi file,
then dividing into train, test, validation sets after data augmentation and feature extraction,
Finally return all sets with composers.
arguments:
conf -- configurations of project.  
'''
def full_data_pipeline(conf):
	np.random.seed(1)
	print('Reading raw data...')
	midis, composers, un_composers = read_dataset(conf['dataset']['raw_path'])
	print('Filter raw data...')
	filter_dataset(midis, composers, conf['data_augmentation']['sample_size'])
	midi_by_composer = []
	for i in range(len(un_composers)):
		midi_by_composer.append([])
	for i, j in zip(midis, composers):
		midi_by_composer[j].append(i)
	x_train = []
	x_test = []
	x_val = []
	y_train = []
	y_test = []
	y_val = []
	x_test_units = []
	y_test_units = []
	x = []
	y = []
	for i in range(len(un_composers)):
		print('Preprocessing composer ', un_composers[i], ' data')
		piano_rolls, features = preprocess_midi_files(conf, midi_by_composer[i])
		x_prelem = combine_features(piano_rolls, features)
		# categorical to have one hot vector.
		y_prelem = to_categorical([i] * len(piano_rolls), num_classes=conf['dataset']['num_class'])
		# Separate Test data from train/validation and keep entire midis/augmented sections to judge both methods.
		x_prelem_train, x_composer_test, y_prelem_train, y_composer_test = train_test_split(x_prelem, y_prelem, test_size=0.15, random_state=1, shuffle = True)
		test_rolls, test_features = uncombine_features(x_composer_test)
		aug_test_rolls, aug_test_features = sample_data(conf, test_rolls, test_features)
		x_composer_test_units = combine_features(aug_test_rolls, aug_test_features)
		y_composer_test_units = to_categorical([i] * len(aug_test_rolls), num_classes=conf['dataset']['num_class'])
		# Separate Validation data from training
		x_composer_train, x_composer_val, y_composer_train, y_composer_val = train_test_split(x_prelem_train, y_prelem_train, test_size=0.15, random_state=1, shuffle = True)
		piano_rolls, features = uncombine_features(x_composer_train)
		aug_piano_rolls, aug_features = sample_data(conf, piano_rolls, features)
		x_composer_train = combine_features(aug_piano_rolls, aug_features)
		y_composer_pure = [i] * len(aug_piano_rolls)
		y_composer_train = to_categorical(y_composer_pure, num_classes=conf['dataset']['num_class'])
		# Validation Data
		piano_rolls_val, features_val = uncombine_features(x_composer_val)
		aug_piano_rolls_val, aug_features_val = sample_data(conf, piano_rolls_val, features_val)
		x_composer_val = combine_features(aug_piano_rolls_val, aug_features_val)
		y_composer_val_pure = [i] * len(aug_piano_rolls_val)
		y_composer_val = to_categorical(y_composer_val_pure, num_classes=conf['dataset']['num_class'])
		# Append results.
		x_train.extend(x_composer_train)
		x_test.extend(x_composer_test)
		x_val.extend(x_composer_val)
		y_train.extend(y_composer_train)
		y_test.extend(y_composer_test)
		y_val.extend(y_composer_val)
		x.extend(x_prelem_train)
		x.extend(x_composer_test)
		y.extend(y_prelem_train)
		y.extend(y_composer_test)
		x_test_units.extend(x_composer_test_units)
		y_test_units.extend(y_composer_test_units)
	return x_train, x_test, x_val, y_train, y_test, y_val, x, y, un_composers, x_test_units, y_test_units
'''
preprocess_midi_files function extracts features and pianorolls
arguments:
conf -- configurations of project.
midis -- midi files.
returns:
piano_rolls -- pianorolls
features -- features of all pianoroll.
'''
def preprocess_midi_files(conf, midis):
	piano_rolls, features = extract_features(conf, midis)
	return piano_rolls, features
'''
prediction_data_pipeline function
arguments:
conf -- configurations of project.
path -- path of midi file.
returns:
x -- augmented pianorolls and features
'''
def prediction_data_pipeline(conf, path):
	midi = load_midi(path)
	if (midi.get_piano_roll(fs=10).shape[1] < conf['data_augmentation']['sample_size']):
		print('Midi file track time too small, at least ', conf['data_augmentation']['sample_size'] / 10 , ' sec is needed')
		exit()
	piano_rolls, features = preprocess_midi_files(conf, [midi])
	aug_piano_rolls, aug_features = sample_prediction_data(conf, piano_rolls, features)
	x = []
	x.append(aug_piano_rolls)
	x.extend(aug_features)
	return x
'''
prediction_data function sampling prediction data return augmented data.
arguments:
conf -- configurations of project.
piano_roll -- pianoroll.
feature -- features of pianoroll.
returns:
x -- augmented pianorolls and features
'''
def prediction_data(conf, piano_roll, feature):
	aug_piano_rolls, aug_features = sample_prediction_data(conf, piano_roll, feature)
	x = []
	x.append(aug_piano_rolls)
	x.extend(aug_features)
	return x
