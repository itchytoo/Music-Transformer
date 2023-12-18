from miditok import REMIPlus
from miditok.data_augmentation import data_augmentation_dataset
import os
from pathlib import Path
import argparse


if __name__ == '__main__':
  # parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--skip_tokenization', action='store_true')
  parser.add_argument('--skip_augmentation', action='store_true')
  args = parser.parse_args()

  # create the tokenizer
  tokenizer = REMIPlus()

  maestro_path = Path('data/maestro-v3.0.0')
  tokenized_path = Path('data/maestro_tokenized')

  if not args.skip_tokenization:
    # tokenize the MAESTRO dataset
    midi_paths = list(maestro_path.glob('**/*.midi'))
    tokenizer.tokenize_midi_dataset(
      midi_paths,
      out_dir=tokenized_path
    )
    print(f'Tokenized MIDI files saved to the {tokenized_path} directory.')

  if not args.skip_augmentation:
    # augment the MAESTRO dataset
    augmented_path = Path('data/maestro_augmented')
    data_augmentation_dataset(data_path=tokenized_path, 
                              tokenizer=tokenizer,
                              nb_octave_offset=3,
                              nb_vel_offset=3,
                              all_offset_combinations=True,
                              out_path=augmented_path)
    print(f'Augmented token files saved to the {augmented_path} directory.')




