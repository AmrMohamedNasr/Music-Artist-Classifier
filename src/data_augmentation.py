import numpy as np

def sample_piano(piano_roll, sample_size, pref_num_samples):
	piano_len = piano_roll.shape[1]
	possible_sample_times = piano_len - sample_size  + 1
	if (pref_num_samples == None):
		pref_num_samples = possible_sample_times
	num_samples = min(possible_sample_times, pref_num_samples)
	sample_times = np.random.choice(possible_sample_times, num_samples, replace = False)
	piano_rolls = []
	for sample_time in sample_times:
		piano_rolls.append(piano_roll[:, sample_time:sample_time + sample_size])
	return piano_rolls

def sample_data(conf, piano_rolls, features):
	aug_piano_rolls = []
	aug_features = []
	for feature in features:
		aug_features.append([])
	for i in range(len(piano_rolls)):
		p = piano_rolls[i]
		aug_rolls = sample_piano(p, conf['data_augmentation']['sample_size'], conf['data_augmentation']['num_samples_per_track'])
		single_feature = []
		for j in range(len(features)):
			single_feature.append([])
			value = features[j][i]
			for k in range(len(aug_rolls)):
				single_feature[j].append(value)
		aug_piano_rolls.extend(aug_rolls)
		for i in range(len(single_feature)):
			aug_features[i].extend(single_feature[i])
	return aug_piano_rolls, aug_features

def sample_prediction_data(conf, piano_rolls, feature):
	aug_piano_rolls = []
	aug_features = []
	for feature in features:
		aug_features.append([])
	for i in range(len(piano_rolls)):
		p = piano_rolls[i]
		aug_rolls = sample_piano(p, conf['data_augmentation']['sample_size'], None)
		single_feature = []
		for j in range(len(features)):
			single_feature.append([])
			value = features[j][i]
			for k in range(len(aug_rolls)):
				single_feature[j].append(value)
		aug_piano_rolls.extend(aug_rolls)
		for i in range(len(single_feature)):
			aug_features[i].extend(single_feature[i])
	return aug_piano_rolls, aug_features
