from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout, LSTM
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def build_pure_cnn(conf):
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

def build_pure_cnn_drop(conf):
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
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Dense(128)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_moderate_cnn_drop(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Conv2D(16, 3)(x0)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Conv2D(32, 3)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = MaxPooling2D()(x)
  x = Conv2D(64, 3)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Conv2D(128, 3)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = MaxPooling2D()(x)
  x = Conv2D(256, 3)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = MaxPooling2D()(x)
  x = Flatten()(x)
  x = Dense(128)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_moderate_cnn_l2(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Conv2D(16, 3, activity_regularizer=l2(conf['model']['parameters']['l2']))(x0)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(32, 3, activity_regularizer=l2(conf['model']['parameters']['l2']))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(64, 3, activity_regularizer=l2(conf['model']['parameters']['l2']))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, 3, activity_regularizer=l2(conf['model']['parameters']['l2']))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(256, 3, activity_regularizer=l2(conf['model']['parameters']['l2']))(x)
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

def build_moderate_cnn(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Conv2D(16, 5)(x0)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(32, 5)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(64, 5)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, 5)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(256, 5)(x)
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

def build_big_cnn(conf):
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
  x = Conv2D(1024, 3)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(2048, 3)(x)
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

def build_rnn(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(128, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers1(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers11(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers11_Dropout(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  #x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(64)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers2(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(64)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers22(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(64)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers3(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers33(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(256)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers4(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(1024, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model

def build_rnn_dense_layers5(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = GRU(256, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(conf['model']['parameters']['dropout'])(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model


def build_rnn_two_lstm(conf):
  x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
  x = Reshape((conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size']))(x0)
  x = Permute((2, 1))(x)
  x = LSTM(128, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'], return_sequences=True)(x)
  x = BatchNormalization()(x)
  x = LSTM(64, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout'])(x)
  x = BatchNormalization()(x)
  x = Dense(conf['dataset']['num_class'])(x)
  x = Activation('softmax')(x)
  model = Model(inputs = x0, outputs = x)
  return model  
def build_hybrid(conf):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(16, 5, activity_regularizer=l2(conf['model']['parameters']['l2'])))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(Conv1D(32, 5, activity_regularizer=l2(conf['model']['parameters']['l2'])))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Conv1D(64, 5, activity_regularizer=l2(conf['model']['parameters']['l2'])))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Flatten())
    rnn_model = Sequential()
    rnn_model.add(Permute((2,1,3), input_shape = (conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1)))
    rnn_model.add(TimeDistributed(cnn_model))
    rnn_model.add(GRU(128, dropout=conf['model']['parameters']['dropout'], recurrent_dropout=conf['model']['parameters']['dropout']))
    rnn_model.add(Dense(conf['dataset']['num_class']))
    rnn_model.add(Activation('softmax'))
    return rnn_model

def build_dummy(conf):
    x0 = Input(shape=(conf['feature_extraction']['max_note'] - conf['feature_extraction']['min_note'] + 1, conf['data_augmentation']['sample_size'], 1))
    x = Flatten()(x0)
    x = Dense(conf['dataset']['num_class'])(x)
    x = Activation('softmax')(x)
    model = Model(inputs = x0, outputs = x)
    return model

def list_models_methods():
	model_builders = {}
	model_builders['cnn'] = build_pure_cnn
	model_builders['cnn_drop'] = build_pure_cnn_drop
	model_builders['mod_cnn'] = build_moderate_cnn
	model_builders['mod_cnn_l2'] = build_moderate_cnn_l2
	model_builders['mod_cnn_drop'] = build_moderate_cnn_drop
	model_builders['big_cnn'] = build_big_cnn
	model_builders['rnn_two_lstm'] = build_rnn_two_lstm
	model_builders['rnn_dense_layers'] = build_rnn_dense_layers
	model_builders['rnn_dense_layers1'] = build_rnn_dense_layers1
	model_builders['rnn_dense_layers2'] = build_rnn_dense_layers2
	model_builders['rnn_dense_layers3'] = build_rnn_dense_layers3
	model_builders['rnn_dense_layers4'] = build_rnn_dense_layers4
	model_builders['rnn_dense_layers33'] = build_rnn_dense_layers33
	model_builders['rnn_dense_layers22'] = build_rnn_dense_layers22
	model_builders['rnn_dense_layers11'] = build_rnn_dense_layers11
	model_builders['rnn_dense_layers5'] = build_rnn_dense_layers5
	model_builders['rnn_dense_layers11_Dropout'] = build_rnn_dense_layers11_Dropout
	model_builders['rnn'] = build_rnn
    model_builders['hybrid'] = build_hybrid
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
    opt = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate'], beta1=parameters['beta_1'], beta2=parameters['beta_2'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    if (conf['model']['tpu']):
        try:
            device_name = os.environ['COLAB_TPU_ADDR']
            TPU_ADDRESS = 'grpc://' + device_name
            print('Found TPU at: {}'.format(TPU_ADDRESS))
        except KeyError:
            print('TPU not found')
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))
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
        EarlyStopping(monitor='val_loss', patience=10)
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
    plt.savefig(conf['model']['type'] + '_train_val_acc.png')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(conf['model']['type'] + '_train_val_loss.png')
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
