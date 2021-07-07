# birdclef-2021

This repository is an attempt to experiment with various model architectures for recognizing birds from their calls, using the birdclef-2021 dataset available in kaggle.

As a first attempt, the original dataset was preprocessed offline to extract 5 minute audio samples and a mel spectogram of those samples were stored as png files in this gcs bucket - `gs://birdclef-2021/data`. It contains `train`, `test` and `val` sub directories each of which has a `metadata.csv` file which contains the list of input files along with corresponding labels (bird name indices) and additional features.
A `labels.csv` file at the top level directory has a mapping from bird names to indices used in the `metadata.csv` files.

To run training:
```
gsutil -q -m cp -R gs://birdclef-2021/data .
python train.py --data_path=./data --ckpt_dir=./ckpt --batch_size=256 --epochs=200 --data_percent=0.1
```

The dataset is large (20G+) and so to evaluate model architectures use a smaller portion of it by specifying `--data_percent` flag to a desired percentage. Stratified sampling is employed.

Adjust `--batch_size` based on the number of GPUs and memory available.

TODO:

1. (In Progress) Create a TF Transform pipeline to convert the audio samples into 2 minute chunks with background noise removed and stored as TF Records in GCS - `transform.py`
1. Add non birdcall sounds to the dataset to have the model learn to identify birdcalls in addition to classifying them.
1. Evaluate a few popular CNN architectures and publish evaluation metrics. Started with NasNetMobile
1. Generate new samples with raw audio data, possibly with some prefiltering to reduce background noise. Then attempt to learn directly using dialated CNSs.
