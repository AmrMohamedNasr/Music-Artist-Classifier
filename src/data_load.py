import os
import pretty_midi
import concurrent.futures
import glob
'''
load_midi function loading midi file.
arguments:
path -- path of midi file.
returns:
midi -- midi data of track.
'''
def load_midi(path):
  midi = pretty_midi.PrettyMIDI(path)
  midi.remove_invalid_notes()
  return midi
'''
read_dataset function reads all dataset 
returns:
midi_files -- midi files.
composers -- composer per each midi file.
unique_composers -- all composers of dataset.
'''
def read_dataset(directory):
  composers = []
  midi_files = []
  unique_composers = []
  composer_tag = 0
  for filename in os.listdir(directory):
      sub_directory = os.path.join(directory, filename)
      unique_composers.append(filename)
      with concurrent.futures.ProcessPoolExecutor() as executor:
        mid_list = glob.glob(os.path.join(sub_directory, "*.mid"))
        com_list = [composer_tag] * len(mid_list)
        mid_list_opened = executor.map(load_midi, mid_list)
        midi_files.extend(mid_list_opened)
        composers.extend(com_list)
      '''
      # Sequential
      for  file in os.listdir(sub_directory):
          if (file.endswith('mid')):
            midi_path = os.path.join(sub_directory, file)
            print(midi_path)
            midi = load_midi(midi_path)
            midi_files.append(midi)
            composers.append(composer_tag)
      '''
      composer_tag = composer_tag + 1
  print('Midi files loaded :', len(midi_files))
  print('Different classes : ', unique_composers)
  return midi_files, composers, unique_composers
'''
filter_dataset function eliminate dataset that doesn't satisfy the sample size in configurations.
'''
def filter_dataset(midis, composers, sample_size):
  for i in range(len(midis), 0, -1):
    if (midis[i - 1].get_piano_roll(fs=10).shape[1] < sample_size):
      midis.pop(i - 1)
      composers.pop(i - 1)