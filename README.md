# birdclef-2021

This repository is an attempt to experiment with various model architectures for recognizing birds from their calls, using the birdclef-2021 dataset available in kaggle.

The dataset is first preprocessed to generate anchor, positive and negative samples for each training sample. The positive and negative samples are picked at random. More than one each of positive and negative samples can exist to ensure their length is at least as much as the anchor sample audio data.

The next stage is a TF Transform task that preprocesses all the audio files including the positive and negative samples, splits them into 2 second chunks with 1 second overlap and then extracts a mel spectogram of those 2 second audio clips, normalize the data using TF Transform and then store them as TF Records.
The dataset is split into train, val and test sets based on a configurable split percentage.

The data is expected to be in some remote storage that is accessible from TF Transform.

#### TODO:

1. Split the dataset prior to TF Transform pipeline to employ Stratified sampling.
2. Split TF Transform into two stages where first one preprocesses the data into 2 second melspectogram chunks and the second one will run a TF Transform op. In between the two, positive and negative samples can be picked by a non parallel task. This will be more efficient as it avoids having to parse the audio data several times.


To preprocess the data:

```
kaggle competitions download -c birdclef-2021 && unzip -d birdclef-2021.zip dataset
gsutil -mq cp -r dataset gs://<your bucket>/data
python preprocess.py --work-dir=gs://<your bucket>/
python transform.py --work-dir=gs://<your bucket>/ --test-percent=0.1 --val-percent=0.1 --region=us-west1 --runner DataflowRunner --project <gcp-project> --temp_location=gs://<your bucket>/tmp --setup_file ./setup.py --worker_zone us-west1-b --machine_type e2-standard-4 --metadata-file=train_metadata_pos_neg.csv
```

To run training:

```
python train.py --data_path=gs://<your bucket> --ckpt_dir=./ckpt --batch_size=256 --epochs=200 [--cache_dir="/path/on/local-storage/"] [--data_percent=0.5]
```

The dataset is large (~1TB+) and so to evaluate model architectures use a smaller portion of it by specifying `--data_percent` flag to a desired percentage.
Use `--cache_dir` to cache datset locally to speed up training after the first epoch.

Adjust `--batch_size` based on the number of GPUs and memory available.

#### TODO:

1. Add non birdcall sounds to the dataset to have the model learn to identify birdcalls in addition to classifying them.
1. Evaluate a few popular CNN architectures and publish evaluation metrics. Started with a stacked dilated CNN.
