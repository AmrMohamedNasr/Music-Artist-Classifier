import numpy as np
'''
sample_piano function gives from each piano_roll many samples
arguments:
sample_size -- size of sample (timesteps adjusted for all samples).
pref_num_samples -- number of samples wanted from a piano_roll,
you can let it gets maximum number of samples by setting this value to null.
piano_roll -- 3d matrix of notes X timesteps X number of tracks(1) 
'''
def sample_piano(piano_roll, sample_size, pref_num_samples):
	piano_len = piano_roll.shape[1]
	possible_sample_times = piano_len - sample_size  + 1
	if (pref_num_samples == None):
		pref_num_samples = possible_sample_times
	num_samples = min(possible_sample_times, pref_num_samples)
	sample_times = np.random.choice(possible_sample_times, num_samples, replace = False)
	piano_rolls = []
	for sample_time in sample_times:
		piano_rolls.append(piano_roll[:, sample_time:sample_time + sample_size, :])
	return piano_rolls
'''
sample_data function returns samples from piano_rolls
arguments:
conf -- configurations to get sample size and number of samples per track.
piano_rolls -- list of pianorolls.
features -- to be added to augmented data.
returns: 
aug_piano_rolls -- augmented pianorolls.
aug_features -- augmented features.  
'''
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
		# TODO i conflicts.
		for i in range(len(single_feature)):
			aug_features[i].extend(single_feature[i])
	return aug_piano_rolls, aug_features
'''
sample_prediction_data function returns samples from predicted piano_rolls
arguments:
conf -- configurations to get sample size.
piano_rolls -- list of pianorolls.
features -- to be added to augmented data.
returns: 
aug_piano_rolls -- augmented pianorolls.
aug_features -- augmented features.  
'''
def sample_prediction_data(conf, piano_rolls, features):
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
