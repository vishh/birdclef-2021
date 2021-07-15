import os
import pandas as pd
import numpy as np
import ast
import multiprocessing as mp
from datetime import datetime as time

from absl import app
from absl import flags

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path','', 'Path to a directory containing training data')
flags.DEFINE_string('ckpt_dir', '', 'Path to a directory expected to contain model checkpoints')
flags.DEFINE_string('logs_dir', '', 'Path to a directory expected to contain model logs')
flags.DEFINE_string('cache_dir', '', 'Path to a directory where model data will be cached to speed up training')
flags.DEFINE_integer('batch_size', 2048, 'Batch Size for training')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train over the dataset')
flags.DEFINE_float('dropout', 0.5, 'Dropout Rate')
flags.DEFINE_float('data_percent', 1, 'Approx percentage of input data to use for training')
flags.DEFINE_integer('layers', 7, 'Number of dialated CNN layers')


feature_spec = {
          'anchor_normalized': tf.io.FixedLenFeature([126, 128], tf.float32),
          'pos_normalized': tf.io.FixedLenFeature([126, 128], tf.float32),
          'neg_normalized': tf.io.FixedLenFeature([126, 128], tf.float32),
          'latitude_normalized': tf.io.FixedLenFeature([1], tf.float32),
          'longitude_normalized': tf.io.FixedLenFeature([1], tf.float32),
          'month_normalized': tf.io.FixedLenFeature([1], tf.float32),
          'label_integerized': tf.io.FixedLenFeature([1], tf.int64),
}

def format_data(elem):
    X = {key: tf.reshape(elem[key], (-1, 1, 1)) for key in ["latitude_normalized", "longitude_normalized", "month_normalized"]}
    for key in ["anchor_normalized", "pos_normalized", "neg_normalized"]:
        X[key] = elem[key]
    y = elem["label_integerized"]
    return X, y

def filter_data(X, y):
    nans = [tf.reduce_all(tf.math.is_nan(X[key])) for key in X]
    return tf.reduce_all(nans)

def get_dataset(base_dir, batch_size, cache_dir, data_percent=1.0):
    files = tf.data.Dataset.list_files(base_dir + "/*")
    num_files = 0
    for f in files:
        num_files += 1
    num_files *= data_percent
    files = files.take(int(num_files))
    dataset = files.apply(tf.data.experimental.parallel_interleave(
         tf.data.TFRecordDataset, cycle_length=mp.cpu_count()))
    dataset = dataset.map(lambda elem: tf.io.parse_single_example(elem, feature_spec), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_data, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_dir != "" :
        dataset = dataset.cache(filename=os.path.join(cache_dir, os.path.basename(base_dir))+".cache")

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main(argv):
    tf.get_logger().setLevel('ERROR')
    train_dir = os.path.join(FLAGS.data_path, "train-dataset")
    val_dir = os.path.join(FLAGS.data_path, "val-dataset")
    test_dir = os.path.join(FLAGS.data_path, 'test-dataset')
    train_dataset = get_dataset(train_dir, FLAGS.batch_size, FLAGS.cache_dir, FLAGS.data_percent)
    test_dataset = get_dataset(test_dir, FLAGS.batch_size, FLAGS.cache_dir)
    val_dataset = get_dataset(val_dir, FLAGS.batch_size, FLAGS.cache_dir)
    X, y = next(iter(train_dataset))
    print([X[key].shape for key in X] + [y.shape])
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
        model = make_or_restore_model(FLAGS.ckpt_dir, X, FLAGS.dropout, FLAGS.layers)
        print(model.summary())

    callbacks = [
                # This callback saves a SavedModel every epoch
                # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(FLAGS.ckpt_dir, "ckpt-{epoch}"), save_freq="epoch"
        ),
#        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        keras.callbacks.TensorBoard(log_dir=FLAGS.logs_dir)
    ]
    
    history = model.fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, use_multiprocessing=False, epochs=FLAGS.epochs, verbose=1)
    
    test_scores = model.evaluate(val_dataset,verbose=2)
    print("Test loss:", test_scores)
    return
    
def make_or_restore_model(checkpoint_dir, X, dropout, layers):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
    checkpoints = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)


    filters = 64
    filter_width=3
    latent_dimensions = 16
    dialation_rates = [2**i for i in range(layers)]
    input_shape = X['anchor_normalized'].shape[1:]
    month = keras.Input(shape=X['month_normalized'].shape[1:], name="month_normalized")
    latitude = keras.Input(shape=X['latitude_normalized'].shape[1:], name="latitude_normalized")
    longitude = keras.Input(shape=X['longitude_normalized'].shape[1:], name="longitude_normalized")

    input = keras.Input(shape=input_shape, name="input")
    x = input
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
    out = keras.layers.Activation('tanh')(out)
    out = keras.layers.GlobalMaxPool1D()(out)
    flat = keras.layers.Flatten()(x)
    concatenatedFeatures = keras.layers.Concatenate(axis = 1)([flat, tf.reshape(month, (-1,1)), tf.reshape(latitude, (-1, 1)), tf.reshape(longitude, (-1,1))])
    concat = keras.layers.Dense(latent_dimensions)(concatenatedFeatures)
    embedding = keras.Model(inputs=[input, month, latitude, longitude], outputs=concat, name="Embedding")
    print(embedding.summary())
    anchor = keras.Input(shape=input_shape, name="anchor_normalized")
    pos = keras.Input(shape=input_shape, name="pos_normalized")
    neg = keras.Input(shape=input_shape, name="neg_normalized")

    distances = DistanceLayer()(
        embedding([anchor, month, latitude, longitude]),
        embedding([pos, month, latitude, longitude]),
        embedding([neg, month, latitude, longitude]),
    )

    inputs = [month, latitude, longitude, anchor, pos, neg]
    siamese = keras.Model(inputs, outputs=distances)

    model = SiameseModel(siamese)
    model.compile(optimizer=keras.optimizers.Adam(0.0001))
    model.build([ip.shape for ip in inputs])
    return model

class DistanceLayer(keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    def call(self, inputs):
        return self.siamese_network(inputs)
        
    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            
            # Storing the gradients of the loss function with respect to the
            # weights/parameters.
            gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
            
            # Applying the gradients on the model using the specified optimizer
            self.optimizer.apply_gradients(
                zip(gradients, self.siamese_network.trainable_weights)
            )
            
            # Let's update and return the training loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}
        
    def test_step(self, data):
        loss = self._compute_loss(data)
        
        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data[0])
        
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss
    
    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    
if __name__ == '__main__':
  app.run(main)

