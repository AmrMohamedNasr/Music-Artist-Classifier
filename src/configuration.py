import yaml

def print_key_value(key, value, prefix):
	print(prefix + key + ":")
	if isinstance(value, dict):
		for k, v in value.items():
			print_key_value(k, v, prefix + '\t')
	elif isinstance(value,list):
		print(prefix + '\t[', end = "")
		not_first = False
		for v in value:
			if not_first:
				print(',', end = " "),
			print(v, end = "")
			not_first = True
		print(']')
	else:
		print(prefix + '\t', end = "")
		print(value)
def print_conf(conf):
	print('-------------------------------------------------------------------------------------')
	print('Configuration: ')
	for section, value in conf.items():
	    print_key_value(section, value, '\t')
	print('-------------------------------------------------------------------------------------')
def read_configuration(path):
	with open(path, 'r') as ymlfile:
	    cfg = yaml.load(ymlfile)
	print_conf(cfg)
	return cfg