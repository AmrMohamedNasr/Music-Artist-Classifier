import numpy as np

def extract_pianoroll(conf, midis):
	pianorolls = []
	for i in midis:
		midi_piano_roll = i.get_piano_roll(fs=10)
		midi_piano_roll = midi_piano_roll[conf['feature_extraction']['min_note']:conf['feature_extraction']['max_note'] + 1,:]
		pianorolls.append(np.reshape(midi_piano_roll, (midi_piano_roll.shape[0], midi_piano_roll.shape[1], 1)))
	return pianorolls
def extraction_functions():
	extraction_functions = {}
	extraction_functions['pianoroll'] = extract_pianoroll
	return extraction_functions
def extract_features(conf, midis):
	extraction_functions_list = extraction_functions()
	pianorolls = extraction_functions_list['pianoroll'](conf, midis)
	print('pianorolls feature has been constructed')
	features = []
	if conf['feature_extraction']['features']:
		for i in conf['feature_extraction']['features']:
			if i in  extraction_functions_list:
				features.append(extraction_functions_list[i](conf, midis))
				print(i, " feature has been constructed")
			else:
				print(i, " is an invalid feature")
				print("Available features are ", extraction_functions_list.keys())
				print("Notice : pianoroll feature is invalid for user usage")
				exit()
	return pianorolls, features