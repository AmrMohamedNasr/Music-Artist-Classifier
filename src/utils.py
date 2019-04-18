from feature_extraction import *
from data_load import *
from keras.utils import to_categorical
from data_augmentation import *
from sklearn.model_selection import train_test_split
def full_data_pipeline(conf):
	np.seed(1)
	midis, composers, un_composers = read_dataset(conf['dataset']['path'])
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
	x = []
	y = []
	for i in range(len(un_composers)):
		piano_rolls, features = preprocess_midi_files(conf, midi_by_composer[i])
		aug_piano_rolls, aug_features = sample_data(conf, piano_rolls, features)
		x_composer = []
		for i in range(len(aug_piano_rolls)):
			x_composer.append([])
			x_composer[i].append(aug_piano_rolls[i])
			for j in range(len(aug_features)):
				x_composer[i].append(aug_features[j][i])
		print('piano_rolls : ', len(aug_piano_rolls), 'aug_features : ', len(aug_features))
		print(len(x_zip_composer))
		y_composer = to_categorical([i] * len(aug_piano_rolls), num_classes=conf['dataset']['num_class'])
		x_composer_train, x_composer_test, y_composer_train, y_composer_test = train_test_split(x_composer, y_composer, test_size=0.15, random_state=1, shuffle = True)
		x_composer_train, x_composer_val, y_composer_train, y_composer_val = train_test_split(x_composer_train, y_composer_train, test_size=0.15, random_state=1, shuffle = True)
		x_train.extend(x_composer_train)
		x_train.extend(x_composer_train)
		x_train.extend(x_composer_train)
		x_train.extend(x_composer_train)
		x_train.extend(x_composer_train)
		x_train.extend(x_composer_train)
	return x_train, x_test, x_val, y_train, y_test, y_val, x, y

def preprocess_midi_files(conf, midis):
	piano_rolls, features = extract_features(conf, midis)
	return piano_rolls, features

def prediction_data_pipeline(conf, path):
	midi = load_midi(path)
	piano_rolls, features = preprocess_midi_files(conf, [midi])
	return features