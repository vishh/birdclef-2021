import os
import tensorflow as tf
import multiprocessing as mp

feature_spec = {
    'audio_normalized': tf.io.FixedLenFeature([501, 128], tf.float32),
    'latitude_normalized': tf.io.FixedLenFeature([1], tf.float32),
    'longitude_normalized': tf.io.FixedLenFeature([1], tf.float32),
    'month_normalized': tf.io.FixedLenFeature([1], tf.float32),
    'label_integerized': tf.io.FixedLenFeature([1], tf.int64),
}

def format_data(elem):
    X = {key: tf.reshape(elem[key], (-1, 1, 1)) for key in ["latitude_normalized", "longitude_normalized", "month_normalized"]}
    X['audio_normalized'] = elem['audio_normalized']
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
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    return dataset.with_options(options)

def get_data(data_path, batch_size, cache_dir, data_percent):
    train_dir = os.path.join(data_path, "train-dataset")
    val_dir = os.path.join(data_path, "val-dataset")
    test_dir = os.path.join(data_path, 'test-dataset')
    with tf. device("cpu:0"):
        train_dataset = get_dataset(train_dir, batch_size, cache_dir, data_percent)
        test_dataset = get_dataset(test_dir, batch_size, cache_dir)
        val_dataset = get_dataset(val_dir, batch_size, cache_dir)
    return train_dataset, val_dataset, test_dataset

