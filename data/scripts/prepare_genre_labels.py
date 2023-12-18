import pandas as pd
import os
from glob import glob

def get_track_ids(root_dir):
    """
    This function reads the track IDs and puts it into a pandas DataFrame.
    Arguments:
        root_dir (string): The path to the root directory containing the tokenized MIDI files.
    Returns:
        A pandas DataFrame containing the track IDs.
    """
    filenames = glob(root_dir + '/**/*.mid', recursive=True) +\
                glob(root_dir + '/**/*.midi', recursive=True)

    hashes = [os.path.basename(filename).split('.')[0] for filename in filenames]
    track_ids = [os.path.basename(os.path.dirname(filename)) for filename in filenames]
    
    df = pd.DataFrame(data={"TrackID": track_ids, "Hash": hashes})
    return df

def get_genres(path):
    """
    This function reads the genre labels and puts it into a pandas DataFrame.
    Arguments:
        path (string): The path to the genre label file.
    Returns:
        A pandas DataFrame containing the genre labels.
    """
    ids = []
    genres = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                [x, y, *_] = line.strip().split("\t") # ignore the minority genre lables
                ids.append(x)
                genres.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
    return genre_df

def main():
    """
    This function creates a pandas DataFrame that maps the file hashes to their parent directories (which represent the TRACK_ID).
    It also adds a column to the df that maps the TRACK_ID to the genre label.
    """
    # get the track IDs
    track_df = get_track_ids('data/lmd_matched')
    # get the genre labels
    genre_df = get_genres('data/msd_tagtraum_cd1.cls')
    # merge the two dfs
    df = pd.merge(track_df, genre_df, on="TrackID")
    # save the df to a csv file
    df.to_csv('data/genre_labels.csv', index=False)

if __name__ == '__main__':
    main()