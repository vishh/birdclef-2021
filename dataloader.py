import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import ast

class BirdClefDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 base_dir,
                 batch_size,
                 num_classes,
                 max_len=None,
                 shuffle=True):
        
        self.df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.df['labels'] = self.df['labels'].apply(lambda s: list(ast.literal_eval(s)))
        self.n = len(self.df)
        if max_len is not None:
            self.n = max_len
        self.n_labels = num_classes
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path, color_mode = "grayscale")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        rows = range(batches.shape[0])
        X_batch = np.asarray([self.__get_input(batches.iloc[row]['filename']) for row in rows])
        X_addtl_batch = np.asarray([[batches.iloc[row]['month'], batches.iloc[row]['latitude'], batches.iloc[row]['longitude']] for row in rows])
        scaler = StandardScaler()
        X_addtl_batch = scaler.fit_transform(X_addtl_batch)
        y_batch = np.asarray([self.__get_output(batches.iloc[row]['labels'][0], self.n_labels) for row in rows])

        return tuple([X_batch, X_addtl_batch]), y_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

def parse_function(image, extra_features, label):
    image_string = tf.io.read_file(image)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=1)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    
    return image, extra_features, label

def get_dataset(base_dir, batch_size, num_classes, max_size=0):
    df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
    df['labels'] = df['labels'].apply(lambda s: list(ast.literal_eval(s)))
    filenames = [os.path.join(base_dir, f) for f in df['filename']]
    labels = np.array([l[0] for l in df['labels']])
    labels = tf.keras.utils.to_categorical(labels)
    extra_features = df[['latitude', 'longitude', 'month']].to_numpy()
    scaler = StandardScaler()
    scaled_extra_features = scaler.fit_transform(extra_features)
    if max_size > 0:
        filenames = filenames[:max_size]
        labels = labels[:max_size]
        scaled_extra_features = scaled_extra_features[:max_size]
    dataset = tf.data.Dataset.from_tensor_slices((filenames, scaled_extra_features, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    return dataset
