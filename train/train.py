import os
import pandas as pd
import numpy as np
import ast

from absl import app
from absl import flags

from dataloader import BirdClefDataGen
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path','', 'Path to a directory containing training data')
flags.DEFINE_string('ckpt_dir', '', 'Path to a directory expected to contain model checkpoints')
flags.DEFINE_string('model_kind', 'basic_cnn', 'kind of model to train. One of "basic_cnn" or "nasnetmobile"')
flags.DEFINE_integer('batch_size', 2048, 'Batch Size for training')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train over the dataset')
flags.DEFINE_float('data_percent', 0.2, 'Percentage of training data to use for training')

def parse_function(x, label):
    image_string = tf.io.read_file(x['input_1'])

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=1)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    
    return {'input_1':image, 'input_2':x['input_2']}, label

def get_dataset(base_dir, batch_size, num_classes, percent=1):
    df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
    df['labels'] = df['labels'].apply(lambda s: list(ast.literal_eval(s)))
    filenames = np.array([os.path.join(base_dir, f) for f in df['filename']])
    labels = np.array([l[0]-1 for l in df['labels']])
    labels_onehot = keras.utils.to_categorical(labels, num_classes=num_classes)
    extra_features = df[['latitude', 'longitude', 'month']].to_numpy()
    scaler = StandardScaler()
    scaled_extra_features = scaler.fit_transform(extra_features)
    if percent < 1:
        skf = StratifiedKFold(shuffle=True, n_splits= int(1/percent), random_state=1)
        _, idxs = next(skf.split(filenames, labels))
        filenames = filenames[idxs]
        labels_onehot = labels_onehot[idxs]
        scaled_extra_features = scaled_extra_features[idxs]
    dataset = tf.data.Dataset.from_tensor_slices(({'input_1': filenames, 'input_2': scaled_extra_features}, labels_onehot))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    return dataset

def main(argv):
    tf.get_logger().setLevel('ERROR')
    trainDir = os.path.join(FLAGS.data_path, 'train')
    valDir = os.path.join(FLAGS.data_path, 'val')
    testDir = os.path.join(FLAGS.data_path, 'test')
    classesDF = pd.read_csv(os.path.join(FLAGS.data_path,"labels.csv"))
    num_classes = classesDF.shape[0]
    train_dataset = get_dataset(trainDir, FLAGS.batch_size, num_classes, percent=FLAGS.data_percent)
    test_dataset = get_dataset(testDir, FLAGS.batch_size, num_classes, percent=FLAGS.data_percent)
    val_dataset = get_dataset(valDir, FLAGS.batch_size, num_classes, percent=FLAGS.data_percent)
    X, y = next(iter(train_dataset))
    print(X['input_1'].shape, X['input_2'].shape, y.shape)

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Prepare a directory to store all the checkpoints.
    os.makedirs(os.path.join(FLAGS.ckpt_dir, FLAGS.model_kind), exist_ok=True)
    os.makedirs(os.path.join("./logs", FLAGS.model_kind), exist_ok=True)
    # Open a strategy scope.
    with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
        model = make_or_restore_model(FLAGS.model_kind, FLAGS.ckpt_dir, X, num_classes)
        print(model.summary())

 #   train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
 #   val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    #keras.utils.plot_model(model, "nasnet.png")
    callbacks = [
                # This callback saves a SavedModel every epoch
                # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(FLAGS.ckpt_dir, FLAGS.model_kind,"ckpt-{epoch}"), save_freq="epoch"
        ),
        keras.callbacks.TensorBoard(log_dir="./logs/{}".format(FLAGS.model_kind))
        
    ]
    
    history = model.fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, use_multiprocessing=False, epochs=FLAGS.epochs, verbose=1)
    
    test_scores = model.evaluate(val_dataset,verbose=2)
    print("Test loss:", test_scores[0])
    print("Test scores:", test_scores)
#    print(test_scores)
    return

def NasnetModel(input_shapes, num_classes, pooling="avg"):
    input_1 = keras.Input(shape=input_shapes[0], name="input_1")
    base = keras.applications.NASNetMobile(
        input_shape=input_shapes[0],
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling=pooling)
    flat = keras.layers.Flatten()(base(input_1))
    input_2 = keras.Input(shape = input_shapes[1], name="input_2")
    concatenatedFeatures = keras.layers.Concatenate(axis = 1)([flat, input_2])
    dense = keras.layers.Dense(32)(concatenatedFeatures)
    outputs = keras.layers.Dense(num_classes, activation=keras.activations.softmax)(dense)
    m = keras.Model(inputs=[input_1, input_2], outputs=outputs, name="nasnet_model")
    return m

def dilated_cnn_block(x, filters, input_shape, layer_idx, dropout):
    x = keras.layers.Conv2D(filters, (3, 3), dilation_rate=layer_idx,
                            padding='same',
                            input_shape=input_shape)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    
    return x
    
def make_or_restore_model(model_kind, checkpoint_dir, X, num_classes):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
    checkpoints = [os.path.join(checkpoint_dir,model_kind, name) for name in os.listdir(os.path.join(checkpoint_dir, model_kind))]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model of kind ", model_kind)
    model = keras.Sequential()
    if model_kind == "nasnetmobile":
        model = NasnetModel((X['input_1'].shape[1:], X['input_2'].shape[1:]), num_classes)
    elif model_kind == "basic_cnn":
        layers = 3
        filters = 32
        dropout = 0.3
        input_1 = keras.Input(shape=X['input_1'].shape[1:], name="input_1")
        x = input_1
        for i in range(1, layers+1):
            x = dilated_cnn_block(x, filters, X['input_1'].shape[1:], i, dropout)
        x = keras.layers.GlobalAveragePooling2D()(x)
        flat = keras.layers.Flatten()(x)
        input_2 = keras.Input(shape=X['input_2'].shape[1:], name="input_2")
        concatenatedFeatures = keras.layers.Concatenate(axis = 1)([flat, input_2])
        dense = keras.layers.Dense(512)(concatenatedFeatures)
        outputs = keras.layers.Dense(num_classes, activation=keras.activations.softmax)(dense)
        model = keras.Model(inputs=[input_1, input_2], outputs=outputs, name="nasnet_model")
        
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy", keras.metrics.CategoricalAccuracy(), keras.metrics.PrecisionAtRecall(0.8)],#, "precision", "recall"],
    )
    return model 

    
    
if __name__ == '__main__':
  app.run(main)

