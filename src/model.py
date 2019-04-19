from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import os

def build_cnn(conf):
	return None

def build_rnn(conf):
	return None

def build_dummy(conf):
	x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
	x = Conv2D(16, 3)(x0)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32, 3)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D()(x)
	x = Conv2D(64, 3)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(128, 3)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D()(x)
	x = Conv2D(256, 3)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(512, 3)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D()(x)
	x = Flatten()(x)
	x = Dense(128)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(conf['dataset']['num_class'])(x)
	x = Activation('softmax')(x)
	model = Model(inputs = x0, outputs = x)
	return model

def list_models_methods():
	model_builders = {}
	model_builders['cnn'] = build_cnn
	model_builders['rnn'] = build_rnn
	model_builders['None'] = build_dummy
	return model_builders

def get_model_path(conf):
	return conf['model']['save_path'] + conf['model']['type'] + '.h5'

def build_model(conf):
	os.makedirs(os.path.dirname(get_model_path(conf)), exist_ok=True)
	list_model = list_models_methods()
	model_name = conf['model']['type']
	if (model_name in list_model):
		model = list_model[model_name](conf)
		print(model_name, ' model has been built')
	else:
		print(model_name, ' is invalid')
		print('Available models are ', list_model.keys())
		exit()
	parameters = conf['model']['parameters']
	opt = Adam(lr=parameters['learning_rate'], beta_1=parameters['beta_1'], beta_2=parameters['beta_2'])
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	try:
		model.load_weights(get_model_path(conf))
		print('Loaded model from file.')
	except:
		print('Unable to load model from file.')
	return model
def train_model(conf, train_x, train_y, val_x, val_y, model):
	parameters = conf['model']['parameters']
	cbs = [
		ModelCheckpoint(get_model_path(conf), monitor='val_loss', save_best_only=True, save_weights_only=True),
		EarlyStopping(monitor='val_loss', patience=5)
	]
	if (len(train_x) == 1):
		history = model.fit(train_x[0], train_y, epochs=parameters['epochs'], validation_data=(val_x[0], val_y), callbacks=cbs, batch_size=parameters['batch_size'], shuffle = True)
	else:
		history = model.fit(train_x, train_y, epochs=parameters['epochs'], validation_data=(val_x, val_y), callbacks=cbs, batch_size=parameters['batch_size'], shuffle = True)
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('train_val_acc.png')
	plt.show()
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('train_val_loss.png')
	plt.show()
def evaluate_model(conf, model, test_X, test_y):
	if (len(test_X) == 1):
		test_loss, test_acc = model.evaluate(test_X[0], test_y, verbose = 0, batch_size = conf['model']['parameters']['batch_size'])
	else:
		test_loss, test_acc = model.evaluate(test_X, test_y, verbose = 0, batch_size = conf['model']['parameters']['batch_size'])
	print('Test loss:', test_loss)
	print('Test accuracy:', test_acc)
def predict_model(conf, model, x):
	if (len(x) == 1):
		return model.predict(x[0], batch_size = conf['model']['parameters']['batch_size'])
	else:
		return model.predict(x, batch_size = conf['model']['parameters']['batch_size'])
