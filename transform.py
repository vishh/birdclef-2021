import csv
import argparse
import datetime
import librosa
import os
from scipy import signal
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
import tensorflow_transform.beam.impl as beam_impl
#from tensorflow_transform.tf_metadata import dataset_schema

import apache_beam as beam
from apache_beam.io import filebasedsource
from apache_beam.options.pipeline_options import PipelineOptions



def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.compute_and_apply_vocabulary(s)
  x_centered_times_y_normalized = x_centered * y_normalized
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }

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
          latitude = line[3]
          longitude = line[4]
          month = int(line[8].split("-")[1])
          f = line[9]
          return [f, latitude, longitude, month, label]

    def load_audio(self, elem):
      if elem is not None:
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
          return [S_db[:, w]]+elem[2:]
      
    def print_elem(self, elem):
      print(elem[0].shape, elem[1:])
      
    def expand(self, p):
      return (p
              | 'Read input file' >> beam.io.ReadFromText(self.metadata, skip_header_lines=1)
              | 'Parse file' >> beam.Map(self.parse_file)
              | 'Load Audio' >> beam.Map(self.load_audio)
              | 'Get MelSpec' >> beam.Map(self.mel_spec)
#              | 'Print' >> beam.Map(self.print_elem)
      )
      # [END dataflow_simple_feature_extraction]
      
if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    '--work-dir',
    required=True,
    help='Directory for staging and working files. '
    'This can be a Google Cloud Storage path.')
  args, pipeline_args = parser.parse_known_args()
  work_dir = args.work_dir
  data_files_dir = os.path.join(work_dir, 'data')
  beam_options = PipelineOptions(pipeline_args, save_main_session=True)

  tft_temp_dir = os.path.join(work_dir, 'tft-temp')
  train_dataset_dir = os.path.join(work_dir, 'train-dataset')
  eval_dataset_dir = os.path.join(work_dir, 'eval-dataset')
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
