from utils import *
from model import *
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import numpy as np

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
	y_train = np.stack(y_train)
	y_val = np.stack(y_val)
	train_model(conf, mod_train_x, y_train, mod_val_x, y_val, model)
def evaluate(conf):
  _, x_test, _, _, y_test, _, _, _, _, x_test_units, y_test_units = full_data_pipeline(conf)
  model = build_model(conf)
  tu_rolls, tu_features = uncombine_features(x_test_units)
  tu_x = []
  tu_x.append(tu_rolls)
  tu_x.extend(tu_features)
  mod_tu_x = []
  for feat in tu_x:
    mod_tu_x.append(np.stack(feat))
  tu_y = np.stack(y_test_units)
  print('Single sample test results : ')
  evaluate_model(conf, model, mod_tu_x, tu_y)
  print('Full track test results : ')
  pianos_s, feats_s = uncombine_features(x_test)
  y_pred = []
  for s in range(len(pianos_s)):
    piano_s = []
    piano_s.append(pianos_s[s])
    feat_s = []
    for ft in feats_s:
      feat_s.append(ft[s])
    x = prediction_data(conf, piano_s, feat_s)
    mod_x = []
    for feat in x:
      mod_x.append(np.stack(feat))
    y = predict_model(conf, model, mod_x)
    y = np.argmax(y, axis = 1)
    votes = np.bincount(y)
    y_pred_s = np.zeros((conf['dataset']['num_class']))
    for i in range(y_pred_s.shape[0]):
      if (i < len(votes)):
        y_pred_s[i] = votes[i]
      else:
        y_pred_s[i] = 0
    y_pred_s = y_pred_s / np.sum(y_pred_s)
    y_pred.append(y_pred_s)
  y_test = np.stack(y_test)
  y_pred = np.stack(y_pred)
  y_test = y_test.astype('float32')
  y_pred = y_pred.astype('float32')
  y_true = K.constant(y_test)
  y_pred = K.constant(y_pred)
  loss = K.categorical_crossentropy(target=y_true, output=y_pred)
  loss = K.eval(loss)
  loss = np.mean(loss)
  acc = categorical_accuracy(y_true, y_pred)
  acc = K.eval(acc)
  acc = np.mean(acc)
  print('Test loss:', loss)
  print('Test accuracy:', acc)