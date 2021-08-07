import os
import pandas as pd
import numpy as np
import ast
import pickle as pkl
from datetime import datetime as time

from absl import app
from absl import flags

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from dataloader import get_data

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path','', 'Path to a directory containing training data')
flags.DEFINE_string('ckpt_dir', '', 'Path to a directory expected to contain model checkpoints')
flags.DEFINE_string('logs_dir', '', 'Path to a directory expected to contain model logs')
flags.DEFINE_string('cache_dir', '', 'Path to a directory where model data will be cached to speed up training')
flags.DEFINE_integer('batch_size', 32, 'Batch Size for training')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train over the dataset')
flags.DEFINE_float('dropout', 0.5, 'Dropout Rate')
flags.DEFINE_float('data_percent', 1, 'Approx percentage of input data to use for training')
flags.DEFINE_integer('layers', 7, 'Number of dialated CNN layers')
flags.DEFINE_bool('train', False, "Training mode")
flags.DEFINE_bool('display_embeddings', True, "Display Embeddings")



def main(argv):
    tf.get_logger().setLevel('ERROR')
    train_dataset, val_dataset, test_dataset = get_data(FLAGS.data_path, FLAGS.batch_size, FLAGS.cache_dir, FLAGS.data_percent)
    X, y = next(iter(train_dataset))
    print([X[key].shape for key in X] + [y.shape])

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        embeddings = make_or_restore_model(FLAGS.ckpt_dir, X, FLAGS.dropout, FLAGS.layers)
        print(embeddings.summary())
    if FLAGS.train:
        checkpoint_path = os.path.join(FLAGS.ckpt_dir, "cp-{epoch:04d}.ckpt")
        callbacks = [
            # This callback saves a SavedModel every epoch
            # We include the current epoch in the folder name.
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_freq="epoch",
                verbose=1,
                save_weights_only=True
            ),
            #keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            keras.callbacks.TensorBoard(log_dir=FLAGS.logs_dir)
        ]
                
        embeddings.save_weights(checkpoint_path.format(epoch=0))
        
        history = embeddings.fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, use_multiprocessing=False, epochs=FLAGS.epochs, verbose=1)
        
        test_scores = embeddings.evaluate(val_dataset,verbose=2)
        print("Test loss:", test_scores)
    if FLAGS.display_embeddings:
        val_embeddings = get_embeddings(val_dataset, embeddings)
        with tf.io.gfile.GFile(os.path.join(FLAGS.data_path, "val_embed.pkl"), "w") as f:
            np.savetxt(f, val_embeddings, delimiter='\t')

        train_embeddings = get_embeddings(train_dataset, embeddings)
        with tf.io.gfile.GFile(os.path.join(FLAGS.data_path, "train_embed.pkl"), "w") as f:
            np.savetxt(f, train_embeddings, delimiter='\t')
    return

def get_embeddings(dataset, model):
    embeddings = []
    classes = []
    for x, y in dataset:
        embed = model.predict(x)
        if len(embeddings) == 0:
            embeddings = embed
            classes = np.array(y)
        else:
            embeddings = np.append(embeddings, model.predict(x), axis=0)
            classes = np.append(classes, y, axis=0)
    y = np.append(embeddings, classes, axis=-1)
    return y

def make_or_restore_model(checkpoint_dir, X, dropout, layers):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    filters = 64
    filter_width=3
    latent_dimensions = 16
    dialation_rates = [2**i for i in range(layers)]
    input_shape = X['audio_normalized'].shape[1:]
    month = keras.Input(shape=X['month_normalized'].shape[1:], name="month_normalized")
    latitude = keras.Input(shape=X['latitude_normalized'].shape[1:], name="latitude_normalized")
    longitude = keras.Input(shape=X['longitude_normalized'].shape[1:], name="longitude_normalized")

    audio = keras.Input(shape=input_shape, name="audio_normalized")
    x = audio
    for dr in dialation_rates:
        x = keras.layers.Conv1D(filters,
                                filter_width,
                                dilation_rate=dr,
                                padding='causal',
                                name="conv1d_dialation_"+str(dr))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    out = keras.layers.Conv1D(32, latent_dimensions, padding="same")(x)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.GlobalMaxPool1D()(out)
    flat = keras.layers.Flatten()(x)
    concatenatedFeatures = keras.layers.Concatenate(axis = 1)([flat, tf.reshape(month, (-1,1)), tf.reshape(latitude, (-1, 1)), tf.reshape(longitude, (-1,1))])
    concat = keras.layers.Dense(2*latent_dimensions)(concatenatedFeatures)
    out = keras.layers.Activation('tanh')(concat)
    out = keras.layers.Dense(latent_dimensions)(out)
    out = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(out)

    embeddings = keras.Model(inputs=[audio, month, latitude, longitude], outputs=out, name="Embedding")
    embeddings.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                       loss=tfa.losses.TripletSemiHardLoss())
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None :
        print("loading weights from checkpoint {}".format(latest))
        embeddings.load_weights(latest)

    return embeddings


if __name__ == '__main__':
  app.run(main)

