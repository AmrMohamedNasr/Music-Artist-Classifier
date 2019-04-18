from train import *
from run import *
from visualize import *
from configuration import *
import argparse
import os
train_mode = 1
run_mode = 2
vis_mode = 3
eva_mode = 4
parser = argparse.ArgumentParser(description='Music classification by artist')
parser.add_argument('--train', '-t', dest='mode', action='store_const', const=train_mode, help = 'Run the model training procedure')
parser.add_argument('--run', '-r', dest='mode', action='store_const', const=run_mode, help = 'Run the classification procedure of a track')
parser.add_argument('--visualize', '-v', dest='mode', action='store_const', const=vis_mode, help = 'Run the visualization of data')
parser.add_argument('--evaluate', '-e', dest='mode', action='store_const', const=eva_mode, help = 'Run the evaluation of the saved model')
parser.add_argument('--configuration_path', '-c', dest='conf_path', default='configuration.yaml', help = 'The path of the configuration yaml file')
parser.add_argument('--midi_path', '-m', dest='midi_path', default=None, help = 'The path of the midi file to be predicted')
parser.set_defaults(mode=None)

if __name__ == "__main__":
	args = parser.parse_args()
	conf_exists = os.path.isfile(args.conf_path)
	if not conf_exists:
		print('Invalid configuration file path')
		exit()
	if (args.mode):
		conf = read_configuration(args.conf_path)
		if (args.mode == train_mode):
			train(conf)
		elif (args.mode == run_mode):
			if (args.midi_path):
				run(conf, args.midi_path)
			else:
				print('No midi file provided for prediction')
		elif (args.mode == vis_mode):
			visualize(conf)
		elif (args.mode == eva_mode):
			evaluate(conf)
	else:
		print('No specification over running mode ! run with -h flag for help')