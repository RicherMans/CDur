# CDur

Repository for the paper "Towards duration robust weakly supervised sound event detection"

Currently for training due to the difficulties of obtaining the training data for DCASE2017/18, the script only supports training and evaluation of the URBAN-SED corpus.
The links to the datasets for training the [DCASE2017](https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars) and [DCASE2018](https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task4/dataset) datasets are provided.

However, all models in the paper (pretrained) are contained in this repository.

## Usage

The scripts provided in this repo can be used to train and evaluate SED models.
In general, all training is done in weakly labeled fashion (WSSED), while evaluation requires strong labels.

The labels use the common DCASE format and are tab separated value files (tsv).
The training labels are required to be in the following format:

```bash
filename    event_labels
a.wav   event1,event2,event3
b.wav   event4
```

The evaluation labels use the following format:

```bash
filename    onset   offset  event_label
c.wav   0.5 4   Speech
c.wav   0.7 8   Cat
c.wav   0.4 4   Dog
```


### Urban-SED {#urban_sed}

To train (and download) CDur on the URBAN-SED corpus, run the following:

```
cd data
# Downloading and/or preparing the urbansed dataset
bash prepare_urbansed.sh
mkdir -p features
# Training features
python3 extract_feature.py flists/urban_sed_train_weak.tsv -o features/urban_sed_train.h5
# Evaluation features
python3 extract_feature.py flists/urban_sed_test_weak.tsv -o features/urban_sed_test.h5
cd ../
pip3 install -r requirements.txt
python3 run.py train_evaluate runconfigs/cdur_urban_sed.yaml  --test_data data/features/urban_sed_test.h5 --test_label data/flists/urban_sed_test_strong.tsv
```

## Reproduce paper results

If you want to just evaluate the results from the paper (here due to the data restrictions only Urban-SED is used).
First prepare the data as seen in the [URBAN-SED](#urban_sed) section.

```bash
python3 run.py evaluate pretrained/URBAN_SED/  --data data/features/urban_sed_test.h5 --label data/flists/urban_sed_test_strong.tsv
```

Which should return something like:
```
Quick Report: 
|               |   f_measure |   precision |   recall |
|---------------|-------------|-------------|----------|
| event_based   |    0.217338 |    0.205556 | 0.233823 |
| segment_based |    0.647505 |    0.697913 | 0.612787 |
| Time Tagging  |    0.775104 |    0.763552 | 0.792407 |
| Clip Tagging  |    0.771375 |    0.80629  | 0.744837 |
```

### DCASE2017

Since the evaluation labels of the DCASE2017 dataset are easily accessible, just run the following script to reproduce the paper results:

```bash
cd data
bash prepare_dcase2017_eval.sh
python3 extract_feature.py flists/dcase2017_eval_weak.tsv -o features/dcase2017_eval.h5
cd ../
python3 run.py evaluate pretrained/DCASE2017/ --data data/features/features/dcase2017_eval.h5 --label data/flists/dcase2017_eval_strong.tsv
```

The result should return something like:

```
Quick Report: 
|               |   f_measure |   precision |   recall |
|---------------|-------------|-------------|----------|
| event_based   |    0.16225  |    0.190996 | 0.14601  |
| segment_based |    0.491504 |    0.559638 | 0.471156 |
| Time Tagging  |    0.547846 |    0.667211 | 0.483353 |
| Clip Tagging  |    0.536513 |    0.692001 | 0.459966 |
```

Note that the results are macro-averaged. The micro-averaged ones can also be found in the logs.
