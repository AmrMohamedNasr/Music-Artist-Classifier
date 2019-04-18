from utils import *
from model import *

def train(conf):
	x_train, _, x_val, y_train, _, y_val, _, _, _, _, _ = full_data_pipeline(conf)
	model = build_model(conf)
	tr_rolls, tr_features = uncombine_features(x_train)
	train_x = []
	train_x.append(tr_rolls)
	train_x.extend(tr_features)
	vr_rolls, vr_features = uncombine_features(x_val)
	val_x = []
	val_x.append(vr_rolls)
	val_x.extend(vr_features)
	mod_train_x = []
	mod_val_x = []
	for feat in train_x:
		mod_train_x.append(np.stack(feat))
	for feat in val_x:
		mod_val_x.append(np.stack(feat))
	train_model(conf, mod_train_x, y_train, mod_val_x, y_val, model)
def evaluate(conf):
	_, x_test, _, _, y_test, _, _, _, _, x_test_units, y_test_units = full_data_pipeline(conf)