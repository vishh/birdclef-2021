#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import ast
import librosa
from sklearn.model_selection import StratifiedKFold
from progress.bar import Bar
import png
import multiprocessing as mp
from tqdm import tqdm

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('base_data_path', './birdclef-2021', 'Path to the base directory that contains all the raw audio data.')
flags.DEFINE_string('preprocess_output_dir', './preprocessed', "Path to the base directory that will contain all pre-processed audio data")

def main(argv):
    trainMetadataDF = pd.read_csv(os.path.join(FLAGS.base_data_path, 'train_metadata.csv'))
    trainSoundscapesDf = pd.read_csv(os.path.join(FLAGS.base_data_path,'train_soundscape_labels.csv'))
    trainMetadataDF['secondary_labels'] = trainMetadataDF['secondary_labels'].apply(lambda s: list(ast.literal_eval(s)))
    classes = {k: i+1 for i, k in enumerate(trainMetadataDF['primary_label'].unique())}
    df = pd.DataFrame({'label':classes.keys(), 'value': classes.values()})
    
    os.makedirs(FLAGS.preprocess_output_dir, exist_ok=True)
    labelsOut = os.path.join(FLAGS.preprocess_output_dir, "labels.csv")
    df.to_csv(labelsOut)

    skf = StratifiedKFold(shuffle=True, n_splits=20, random_state=1)
    for train_index, test_index in skf.split(trainMetadataDF['filename'], trainMetadataDF['primary_label']):
        train_val_df = trainMetadataDF.iloc[train_index]
        test_df = trainMetadataDF.iloc[test_index]
        break
    for train_index, val_index in skf.split(train_val_df['filename'], train_val_df['primary_label']):
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]
        break
    print(train_df.shape, val_df.shape, test_df.shape)

    sr=32000
    for df, folder in [(train_df, os.path.join(FLAGS.preprocess_output_dir,'train')), (val_df, os.path.join(FLAGS.preprocess_output_dir,'val')), (test_df, os.path.join(FLAGS.preprocess_output_dir,'test'))]:
        os.makedirs(folder, exist_ok=True)
        print("Processing {}".format(folder))
        bar = Bar('Processing', max=df.shape[0])

        metadata_dfs = []
        for idx in tqdm(range(df.shape[0])):
            out = process_row(df,idx,FLAGS.base_data_path,folder,classes)
            if out is not None:
                metadata_dfs.append(out)

        metadata_df = pd.concat(metadata_dfs)
        metadata_df.to_csv(os.path.join(folder, 'metadata.csv'))
        bar.finish()

def process_row(df, idx, base_path, output_path, classes, sr=32000):
    row = df.iloc[idx]
    batches = get_batches(row, base_path, sr)            
    month = int(row['date'].split('-')[1])
    labels = [classes[i] for i in [row['primary_label']]+row['secondary_labels'] if i in classes]
    latitude = row['latitude']
    longitude = row['longitude']
    metadata_dfs = []
    for bi, b in enumerate(batches):
        if b.sum() == 0:
            continue
        filename = os.path.join(output_path,
                                "{}_{}_{}.png".format(
                                    row['primary_label'], os.path.splitext(row['filename'])[0],bi))
        if not os.path.exists(filename):
            png.from_array(b, mode="L").save(filename)
        metadata_dfs.append(
            pd.DataFrame({
                'row_id': "{}_{}_{}".format(row['primary_label'], idx, bi),'labels':[labels], 'filename':filename, 'latitude':latitude, 'longitude':longitude, 'month':month}))

    if len(metadata_dfs) == 0 :
        return None
    return pd.concat(metadata_dfs)

def get_mel_spec(x, sr):
    melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000 
    }
    M = librosa.feature.melspectrogram(x, sr=sr, **melspectrogram_parameters)
    M_db = librosa.power_to_db(M, ref=np.max)
    return M_db

def get_batches(df, base_path, sr=32000):
    shortAudioPath = os.path.join(base_path, 'train_short_audio')
    audio_data = os.path.join(shortAudioPath, df['primary_label'], df['filename'])
    x, sr = librosa.load(audio_data, sr=sr, mono=True)
    batchLen = 5*sr
    result = [] 
    for i in range(len(x)//batchLen):
        b = x[i*batchLen:(i+1)*batchLen]
        if len(b) < batchLen:
            break
        mel = get_mel_spec(b, sr)
        result.append(mel)
    eps = 1e-6
    X = np.array(result)
    # Standardize
    X = X - np.mean(X)
    std = X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < _min] = _min
        V[V > _max] = _max
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
    



if __name__ == '__main__':
  app.run(main)
