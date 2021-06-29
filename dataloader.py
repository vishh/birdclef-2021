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

