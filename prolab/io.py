# -*- coding: utf-8 -*-
'''
Script: io.py
Description: functions to read and write data.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python version: 3.11

Dependencies:
    - numpy
    - pandas
    - seaborn

Usage:
    python example_analysis.py
'''

# %% Dependencies

# Libraries
from pathlib import Path
import pandas as pd
from parse import parse


# %% Function to read files

def read_files(path, format_string, sample_pattern, ref_pattern):
    '''
    Read files from WPI measurements (all files in a directory).

    Parameters
    ----------
    path : str
        Path to the folder where the files are located.
    format_string : str
        A parsing pattern string (template) used to extract information
        from file names. Should contain at least `pattern` and `id` strings.

        For example: '{pattern}{id:d}{replicate}_{campaign}'

        - `pattern`  : informs whether it's sample or reference (mandatory)
        - `id`         : integer identifier (mandatory)
        - `replicate`  : replicate label (optional)
        - `campaign`   : campaign name (optional)

        It would be used to deal with filenames such as `'station01a_c01'`.

        Read the documentation of the `parse` module for further details.
    sample_pattern : str
        String used as pattern to identify sample measurements.
    ref_pattern : str
        String used as pattern to identify reference measurements.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame indexed by the filenames. The columns present the
        variables parsed from the filename, and the values by wavelength.
    '''

    # Check format string
    if 'id' not in format_string:
        raise ValueError('The format string must contain an `id` string')
    elif 'pattern' not in format_string:
        raise ValueError('The format string must contain a `pattern` string')

    # Update path
    path = Path(path)

    # Create dict to receive curves
    curve_dict = {}

    # Iterate over files
    for file in path.iterdir():

        # Open file
        raw = pd.read_table(filepath_or_buffer=file,
                            index_col=0,
                            skiprows=44,
                            header=None,
                            encoding='latin1',
                            engine='python')

        # Add to list
        curve_dict[file.stem] = raw.dropna().mean(axis=1)

    # Create dataframe
    curves = pd.DataFrame(curve_dict).transpose()

    # Check for NaN
    if curves.isna().any().any():
        print('Warning: there are NaNs in the data')

    # Parse names
    parsed = curves.index.map(lambda x: parse(format_string, x).named)

    # Create dataframe with metadata
    meta = pd.DataFrame(parsed.tolist(), index=curves.index)

    # Create column to identify reference measurements
    meta['is_ref'] = [ref_pattern in name for name in meta.index]

    # Final dataframe
    df = meta.merge(curves, left_index=True, right_index=True)
    df.index.name = 'key'

    # Return data
    return df
