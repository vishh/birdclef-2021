import os
import pandas as pd
import numpy as np

from absl import app
from absl import flags

from dataloader import BirdClefDataGen, get_dataset

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path','', 'Path to a directory containing training data')
flags.DEFINE_string('ckpt_dir', '', 'Path to a directory expected to contain model checkpoints')
flags.DEFINE_integer('batch_size', 16, 'Batch Size for training')

def main(argv):
    trainDir = os.path.join(FLAGS.data_path, 'train')
    valDir = os.path.join(FLAGS.data_path, 'val')
    testDir = os.path.join(FLAGS.data_path, 'test')
    classesDF = pd.read_csv(os.path.join(FLAGS.data_path,"labels.csv"))
    num_classes = classesDF.shape[0]
    train_dataset = get_dataset(trainDir, FLAGS.batch_size, num_classes, max_size=FLAGS.batch_size)
    val_dataset = get_dataset(valDir, FLAGS.batch_size, num_classes, max_size=FLAGS.batch_size)
    X, y = next(iter(train_dataset))
    model = NasnetModel((X[0].shape[1:], X[1].shape[1:]), num_classes)
    print(model.summary())
    
    #tf.keras.utils.plot_model(model, "nasnet.png")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy", "precision", "recall"],
    )
    
    history = model.fit(train_dataset, validation_data=val_dataset, use_multiprocessing=False, workers=1, epochs=2)
    
    test_scores = model.evaluate(val_dataset,verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print(test_scores)
    return

def NasnetModel(input_shapes, num_classes, pooling="avg"):
    inputs = tf.keras.Input(shape=input_shapes[0])
    base = tf.keras.applications.NASNetMobile(
        input_shape=input_shapes[0],
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling=pooling)
    flat = tf.keras.layers.Flatten()(base(inputs))
    otherInp = tf.keras.Input(shape = input_shapes[1])
    concatenatedFeatures = tf.keras.layers.Concatenate(axis = 1)([flat, otherInp])
    dense = tf.keras.layers.Dense(512)(concatenatedFeatures)
    outputs = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)(dense)
    m = tf.keras.Model(inputs=[inputs, otherInp], outputs=outputs, name="nasnet_model")
    return m
    
    


if __name__ == '__main__':
  app.run(main)

