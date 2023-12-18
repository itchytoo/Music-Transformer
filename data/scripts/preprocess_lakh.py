import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os
from tqdm import tqdm
from anticipation.convert import midi_to_compound
from anticipation.config import PREPROC_WORKERS

def convert_midi(filename, debug=False):
    """
    Convert a MIDI file to compound format and save it to a new directory.

    Args:
        filename (str): The path to the MIDI file.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        int: 0 if the conversion is successful, 1 otherwise.
    """
    try:
        tokens = midi_to_compound(filename, debug=debug)
    except Exception:
        if debug:
            print('Failed to process: ', filename)
            print(traceback.format_exc())
        return 1

    # get the first hex digit of the hash
    hex_digit = os.path.basename(filename)[0] 
    output_path = os.path.join('data', 'lmd_processed', hex_digit, os.path.basename(filename))

    # save the compound tokens to the output file
    with open(f"{output_path}.compound.txt", 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))
    return 0

def main(args):
    """
    Preprocess MIDI files in a directory.

    Args:
        args (Namespace): The command-line arguments.

    Returns:
        None
    """
    # prepare the output directory
    if not os.path.exists('data/lmd_processed'):
        os.makedirs('data/lmd_processed')
    HEX = '0123456789abcdef'
    for c in HEX:
        if not os.path.exists(f'data/lmd_processed/{c}'):
            os.makedirs(f'data/lmd_processed/{c}')

    # get all the MIDI files in the directory
    filenames = glob(args.dir + '/**/*.mid', recursive=True) +\
                glob(args.dir + '/**/*.midi', recursive=True)

    print(f'Preprocessing {len(filenames)} files with {PREPROC_WORKERS} workers')

    # convert the MIDI files to compound format
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        results = list(tqdm(executor.map(convert_midi, filenames), desc='Preprocess', total=len(filenames)))

    # print the number of successfully processed files
    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully processed {len(filenames) - sum(results)} files (discarded {discards}%)')

if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    main(parser.parse_args())
