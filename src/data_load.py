import os
import pretty_midi
import concurrent.futures
import glob

def load_midi(path):
  midi = pretty_midi.PrettyMIDI(path)
  midi.remove_invalid_notes()
  return midi

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