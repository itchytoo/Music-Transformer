# Aligning Music Transformers with RLHF

# Music Transformer Repository

This repository contains code for preprocessing the Lakh MIDI dataset and the Maestro dataset. It also includes code for pretraining a PaLM (Piano Language Model) on the Maestro dataset. Additionally, it provides code for training a genre classification model. Finally, it offers code to fine-tune either the Anticipatory Transformer by Thickstun or the PaLM model using the genre classification reward model.

## Setup

1. Run `pip install -r requirements.txt` to install the dependencies.
2. Run `git clone https://github.com/jthickstun/anticipation.git` to download the Anticipitory Transformer repo.
3. Run `pip install anticipation` to install the anticipation package

## PaLM Pretraining Instructions

1. Run `python data/scripts/download_dataset.py --dataset maestro` to download the Maestro dataset.
2. Run `python data/scripts/preprocess_maestro.py` to augment and tokenize the dataset.
3. Run `python train/train.py` to train the PaLM model on the dataset.

## Classifier Training Instructions

1. Run `python data/scripts/download_dataset.py --dataset lakh` to download the Lakh MIDI dataset
2. Run `python data/scripts/preprocess_lakh.py data/lmd_matched` to preprocess the dataset. This step takes ~40 minutes.
3. Run `python data/scripts/tokenize_lakh.py data/lmd_processed` to tokenize the dataset. 
4. Run `python data/scripts/prepare_genre_labels.py` to prepare the genre labels.
5. Run  `python train/train_classifier.py` to train the classifier model on the dataset

## RLHF Fine Tuning Instructions

In Progress.



