import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os
from tqdm import tqdm
from anticipation.tokenize import tokenize
from anticipation.config import PREPROC_WORKERS

def tokenize_file(filename, debug=False):
    """
    """
    try:
        output_path = filename.replace('lmd_processed', 'lmd_tokenized').replace('.mid.compound.txt', '.txt')
        tokenize([filename], output_path, 1)

    except Exception:
        if debug:
            print('Failed to tokenize: ', filename)
            print(traceback.format_exc())
        return 1

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
    if not os.path.exists('data/lmd_tokenized'):
        os.makedirs('data/lmd_tokenized')
    HEX = '0123456789abcdef'
    for c in HEX:
        if not os.path.exists(f'data/lmd_tokenized/{c}'):
            os.makedirs(f'data/lmd_tokenized/{c}')

    # get all the compound.txt files in the directory
    filenames = glob(args.dir + '/**/*.compound.txt', recursive=True) 

    print(f'Tokenizing {len(filenames)} files with {PREPROC_WORKERS} workers')

    # convert the MIDI files to compound format
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        results = list(tqdm(executor.map(tokenize_file, filenames), desc='Tokenize', total=len(filenames)))

    # print the number of successfully tokenized files
    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully tokenized {len(filenames) - sum(results)} files (discarded {discards}%)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a MIDI dataset')
    parser.add_argument('dir', help='directory containing .compound.txt files for training')
    main(parser.parse_args())
