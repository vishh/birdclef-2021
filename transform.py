import csv
import argparse
import datetime
import librosa
import os
import random
from scipy import signal
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder


import apache_beam as beam
from apache_beam.io import filebasedsource
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions

# [START dataflow_molecules_simple_feature_extraction]
class SimpleFeatureExtraction(beam.PTransform):
    """The feature extraction (element-wise transformations).
  We create a `PTransform` class. This `PTransform` is a bundle of
  transformations that can be applied to any other pipeline as a step.
  We'll extract all the raw features here. Due to the nature of `PTransform`s,
  we can only do element-wise transformations here. Anything that requires a
  full-pass of the data (such as feature scaling) has to be done with
  tf.Transform.
  """
    def __init__(self, dir):
      super(SimpleFeatureExtraction, self).__init__()
      self.metadata = os.path.join(dir, "train_metadata.csv")
      self.train_data_dir = os.path.join(dir, "train_short_audio")
      
    def parse_file(self, element):
      for line in csv.reader([element], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if float(line[-3]) >= 2.5:
          label = line[0]
          latitude = float(line[3])
          longitude = float(line[4])
          month = int(line[8].split("-")[1])
          f = line[9]
          return [[f, latitude, longitude, month, label]]

    def load_audio(self, elem):
      filepath = os.path.join(self.train_data_dir, elem[-1], elem[0])
      with tf.io.gfile.GFile(filepath, 'rb') as fp:
        x, sr = librosa.load(fp, sr=32000, mono=True)
        return [x, sr]+elem[1:]

    def mel_spec(self, elem):
      sr = elem[1]
      melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 150,
        "fmax": 15000,
        "hop_length": sr, # one second
        "win_length" : 2*sr, # 2 seconds resulting in 50% overlap
        "n_fft": 2*sr,
      }
      
      sos = signal.butter(10, [150,15000], 'bandpass', fs=sr, output='sos')
      x = signal.sosfilt(sos, elem[0])
      if len(x) >= 2*sr:
        S = librosa.feature.melspectrogram(x, sr=sr,  **melspectrogram_parameters)
        S_db = librosa.power_to_db(S, ref=np.max)
        for w in range(S_db.shape[1]):
          if S_db[:, w].shape[0] > 0:
            return {'audio': S_db[:, w][0], 'latitude': elem[2], 'longitude': elem[3], 'month': elem[4], 'label': elem[5]}
          
    def print_fn(self, elem):
      print(type(elem))
      return elem
    
    def expand(self, p):
      return (p
              | 'Read input file' >> beam.io.ReadFromText(self.metadata, skip_header_lines=1)
              | 'Parse file' >> beam.FlatMap(self.parse_file)
              | 'Load Audio' >> beam.Map(self.load_audio)
              | 'Get MelSpec' >> beam.Map(self.mel_spec)
#              | 'Print' >> beam.Map(self.print_fn)
      )
      # [END dataflow_simple_feature_extraction]

# [START dataflow_normalize_inputs]
def normalize_inputs(inputs):
    """Preprocessing function for tf.Transform (full-pass transformations).
  Here we will do any preprocessing that requires a full-pass of the dataset.
  It takes as inputs the preprocessed data from the `PTransform` we specify, in
  this case `SimpleFeatureExtraction`.
  Common operations might be scaling values to 0-1, getting the minimum or
  maximum value of a certain field, creating a vocabulary for a string field.
  There are two main types of transformations supported by tf.Transform, for
  more information, check the following modules:
    - analyzers: tensorflow_transform.analyzers.py
    - mappers:   tensorflow_transform.mappers.py
  Any transformation done in tf.Transform will be embedded into the TensorFlow
  model itself.
  """
    if inputs is not None:
      label_integerized = tft.compute_and_apply_vocabulary(inputs['label'])
      return {
        # Scale the input features for normalization
        'audio_normalized': tft.scale_to_0_1(inputs['audio']),
        'latitude_normalized': tft.scale_to_0_1(inputs['latitude']),
        'longitude_normalized': tft.scale_to_0_1(inputs['longitude']),
        'month_normalized': tft.scale_to_0_1(inputs['month']),
        'label_integerized': label_integerized
      }
  # [END dataflow_normalize_inputs]

def run_pipeline(work_dir, beam_options, test_percent, val_percent):
  data_files_dir = os.path.join(work_dir, 'data')
  tft_temp_dir = os.path.join(work_dir, 'tft-temp')
  labels_dir = os.path.join(work_dir, 'labels')
  train_dataset_dir = os.path.join(work_dir, 'train-dataset')
  val_dataset_dir = os.path.join(work_dir, 'val-dataset')
  test_dataset_dir = os.path.join(work_dir, 'test-dataset')
  transform_fn_dir = os.path.join(work_dir, transform_fn_io.TRANSFORM_FN_DIR)
  if tf.io.gfile.exists(transform_fn_dir):
    tf.io.gfile.rmtree(transform_fn_dir)
    
    # [START dataflow_create_pipeline]
    # Build and run a Beam Pipeline
  with beam.Pipeline(options=beam_options) as p, \
       beam_impl.Context(temp_dir=tft_temp_dir):
    # [START dataflow_feature_extraction]
    # Transform and validate the input data matches the input schema
    dataset = (
      p
      | 'Feature extraction' >> SimpleFeatureExtraction(data_files_dir)
      # [END dataflow_feature_extraction]
      )
    # [END dataflow_validate_inputs]
 
    # [START dataflow_molecules_analyze_and_transform_dataset]
    # Apply the tf.Transform preprocessing_fn
    input_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec({
        'audio': tf.io.FixedLenFeature([], tf.float32),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'month': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
      }))

    dataset_and_metadata, transform_fn = (
      (dataset, input_metadata)
      | 'Feature scaling' >> beam_impl.AnalyzeAndTransformDataset(normalize_inputs)
    )
    dataset, metadata = dataset_and_metadata

    # [START dataflow_split_to_train_and_eval_datasets]
    # Split the dataset into a training set and an test set
    assert 0 < test_percent < 100, 'test_percent must in the range (0-100)'
    train_dataset, test_dataset = (
      dataset
      | 'Split test/train dataset' >> beam.Partition(
        lambda elem, _: int(random.uniform(0, 100) < test_percent), 2))
    assert 0 < val_percent < 100, 'val_percent must be in the range (0-100)'
    train_dataset, val_dataset = (
      train_dataset
      | 'Split train/val dataset' >> beam.Partition(
        lambda elem, _: int(random.uniform(0, 100) < val_percent), 2))
    # [END dataflow_split_to_train_and_eval_datasets]

    # [START dataflow_write_tfrecords]
    # Write the datasets as TFRecords

    coder = example_proto_coder.ExampleProtoCoder(metadata.schema)
    
    train_dataset_prefix = os.path.join(train_dataset_dir, 'part')
    _ = (
      train_dataset
      | 'Write train dataset' >> tfrecordio.WriteToTFRecord(
        train_dataset_prefix, coder))
    
    val_dataset_prefix = os.path.join(val_dataset_dir, 'part')
    _ = (
      val_dataset
      | 'Write val dataset' >> tfrecordio.WriteToTFRecord(
        val_dataset_prefix, coder))

    test_dataset_prefix = os.path.join(test_dataset_dir, 'part')
    _ = (
      test_dataset
      | 'Write test dataset' >> tfrecordio.WriteToTFRecord(
        test_dataset_prefix, coder))

    # Write the transform_fn
    _ = (
      transform_fn
      | 'Write transformFn' >> transform_fn_io.WriteTransformFn(work_dir))
    # [END dataflow_write_tfrecords]


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    '--work-dir',
    required=True,
    help='Directory for staging and working files. '
    'This can be a Google Cloud Storage path.')
  parser.add_argument(
    '--test-percent',
    required=True,
    help='Percentage of samples to set aside for test set')
  parser.add_argument(
    '--val-percent',
    required=True,
    help='Percentage of samples to set aside for val set')

  args, pipeline_args = parser.parse_known_args()
  work_dir = args.work_dir

  beam_options = PipelineOptions(pipeline_args, save_main_session=True)

  run_pipeline(work_dir, beam_options, float(args.test_percent), float(args.val_percent))
