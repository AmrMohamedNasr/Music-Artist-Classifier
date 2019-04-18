from data_load import *
from utils import *
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import math

def tracks_per_composer(composers, un_composers):
	c = Counter(composers)
	plt.bar(range(len(c)), list(c.values()), align='center')
	keys = [un_composers[i] for i in c.keys()]
	plt.xticks(range(len(un_composers)), keys, rotation=45)
	plt.ylabel('# of tracks')
	plt.xlabel('Composers')
	plt.title('tracks per composer')
	plt.tight_layout()
	plt.savefig('tracks_per_composer.jpg')
	plt.show()

def total_track_time(midis, composers, un_composers):
	midi_files = {}
	for i in un_composers:
		midi_files[i] = 0
	for i, j in zip(midis, composers):
		midi_files[un_composers[j]] += i.get_piano_roll(fs=10).shape[1]
	plt.bar(range(len(midi_files)), list(midi_files.values()), align='center')
	plt.xticks(range(len(midi_files)), list(midi_files.keys()), rotation=45)
	plt.ylabel('Total time of tracks')
	plt.xlabel('Composers')
	plt.title('Total track time per composer')
	plt.tight_layout()
	plt.savefig('total_time_per_composer.jpg')
	plt.show()
def hist_track_time(midis):
	track_time = [i.get_piano_roll(fs=10).shape[1] for i in midis]
	bins = np.linspace(math.ceil(min(track_time)), 
                   math.floor(max(track_time)),
                   50)
	plt.hist(track_time, bins=bins)
	plt.title('Track time histogram')
	plt.xlabel('track length')
	plt.ylabel('count')
	plt.tight_layout()
	plt.savefig('track_time_histogram.jpg')
	plt.show()
def single_track_time(midis, composers, un_composers):
	midi_files = {}
	for i in un_composers:
		midi_files[i] = []
	for i, j in zip(midis, composers):
		midi_files[un_composers[j]].append(i.get_piano_roll(fs=10).shape[1])
	for i in un_composers:
		plt.hist(midi_files[i], alpha=0.5, label=i)
	plt.legend(loc='upper right')
	plt.title('Track time histogram')
	plt.xlabel('track length')
	plt.ylabel('count')
	plt.tight_layout()
	plt.savefig('multi_track_time_histogram.jpg')
	plt.show()
def plot_bar_track_time_set(x, y, label, un_composers):
	midi_files = {}
	for i in un_composers:
		midi_files[i] = 0
	for i, j in zip(x, y):
		index = np.where(j==1)[0][0]
		midi_files[un_composers[index]] += i[0].shape[1]
	plt.bar(range(len(midi_files)), list(midi_files.values()), align='center')
	plt.xticks(range(len(midi_files)), list(midi_files.keys()), rotation=45)
	plt.ylabel('Total time of tracks')
	plt.xlabel('Composers')
	plt.title('Total track time per composer in ' + label + ' set')
	plt.tight_layout()
	plt.savefig(label + '_total_time_per_composer.jpg')
	plt.show()

def plot_bar_sample_number_set(x, y, label, un_composers):
	midi_files = {}
	for i in un_composers:
		midi_files[i] = 0
	for i, j in zip(x, y):
		index = np.where(j==1)[0][0]
		midi_files[un_composers[index]] += 1
	plt.bar(range(len(midi_files)), list(midi_files.values()), align='center')
	plt.xticks(range(len(midi_files)), list(midi_files.keys()), rotation=45)
	plt.ylabel('# of samples')
	plt.xlabel('Composers')
	plt.title('# of samples per composer in ' + label + ' set')
	plt.tight_layout()
	plt.savefig(label + '_samples_per_composer.jpg')
	plt.show()

def visualize(conf):
	midis, composers, un_composers = read_dataset(conf['dataset']['raw_path'])
	x_train, x_test, x_val, y_train, y_test, y_val, x, y, un_composers, x_test_units, y_test_units = full_data_pipeline(conf)
	tracks_per_composer(composers, un_composers)
	total_track_time(midis, composers, un_composers)
	hist_track_time(midis)
	single_track_time(midis, composers, un_composers)
	plot_bar_track_time_set(x, y, 'total', un_composers)
	plot_bar_track_time_set(x_train, y_train, 'train', un_composers)
	plot_bar_track_time_set(x_test, y_test, 'test', un_composers)
	plot_bar_track_time_set(x_val, y_val, 'val', un_composers)
	plot_bar_track_time_set(x_test_units, y_test_units, 'test_units', un_composers)
	plot_bar_sample_number_set(x, y, 'total', un_composers)
	plot_bar_sample_number_set(x_train, y_train, 'train', un_composers)
	plot_bar_sample_number_set(x_test, y_test, 'test', un_composers)
	plot_bar_sample_number_set(x_val, y_val, 'val', un_composers)
	plot_bar_sample_number_set(x_test_units, y_test_units, 'test_units', un_composers)