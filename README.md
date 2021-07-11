# birdclef-2021

This repository is an attempt to experiment with various model architectures for recognizing birds from their calls, using the birdclef-2021 dataset available in kaggle.

As a first attempt, the original dataset was preprocessed offline to extract 5 minute audio samples and a mel spectogram of those samples were stored as png files in this gcs bucket - `gs://birdclef/data`.
There are TFRecords for `train`, `val` and `test` sets.

To preprocess the data:
```
kaggle competitions download -c birdclef-2021 && unzip -d birdclef-2021.zip dataset
gsutil -mq cp -r dataset gs://<your bucket>/data
python preprocess.py --work-dir=gs://<your bucket>/
python transform.py --work-dir=gs://<your bucket>/ --test-percent=0.1 --val-percent=0.1 --region=us-west1 --runner DataflowRunner --project <gcp-project> --temp_location=gs://<your bucket>/tmp --setup_file ./setup.py --worker_zone us-west1-b --machine_type e2-standard-4 --metadata-file=train_metadata_pos_neg.csv
```

To run training:
```
python train.py --data_path=gs://<your bucket> --ckpt_dir=./ckpt --batch_size=256 --epochs=200 --data_percent=0.1
```

The dataset is large (20G+) and so to evaluate model architectures use a smaller portion of it by specifying `--data_percent` flag to a desired percentage. Stratified sampling is employed.

Adjust `--batch_size` based on the number of GPUs and memory available.

TODO:

1. (In Progress) Create a TF Transform pipeline to convert the audio samples into 2 minute chunks with background noise removed and stored as TF Records in GCS - `transform.py`
1. Add non birdcall sounds to the dataset to have the model learn to identify birdcalls in addition to classifying them.
1. Evaluate a few popular CNN architectures and publish evaluation metrics. Started with NasNetMobile
1. Generate new samples with raw audio data, possibly with some prefiltering to reduce background noise. Then attempt to learn directly using dialated CNSs.
