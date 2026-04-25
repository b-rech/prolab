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

def read_files(path, instrument, format_string, sample_pattern,
               ref_pattern, decimal='.', force_unique=False):
    '''
    Read files from WPI measurements (all files in a directory).

    Parameters
    ----------
    path : str
        Path to the folder where the files are located.
    instrument : str
        Name of the instrument that generated the files.

        Supported instruments: `wpi` and `shimadzu`.

    format_string : str
        A parsing pattern string (template) used to extract information
        from file names. Should contain at least `pattern` and `id` strings.

        For example: `{pattern}{id:d}{replicate}_{campaign}`

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
        variables parsed from the filename, and the wavelengths.
    '''

    # -------------------------------------------------------------------------
    # Initial checks
    # -------------------------------------------------------------------------

    # Check format string
    if 'id' not in format_string:
        raise ValueError('The format string must contain an "id" string')
    elif 'pattern' not in format_string:
        raise ValueError('The format string must contain a "pattern" string')

    # Update path
    path = Path(path)

    # -------------------------------------------------------------------------
    # Specific processing of WPI data
    # -------------------------------------------------------------------------

    if instrument == 'wpi':

        # Create dict to receive curves
        curve_dict = {}

        # Iterate over files
        for file in path.iterdir():

            # Open file
            raw = pd.read_table(filepath_or_buffer=file,
                                decimal=decimal,
                                index_col=0,
                                skiprows=44,
                                header=None,
                                encoding='latin1',
                                engine='python')

            # Add to list
            curve_dict[file.stem] = raw.dropna().mean(axis=1)

        # Create dataframe
        curves = pd.DataFrame(curve_dict).transpose()


    # -------------------------------------------------------------------------
    # Specific processing of Shimadzu data
    # -------------------------------------------------------------------------
    if instrument == 'shimadzu':

        # Create list to store tables
        table_list = []

        # Iterate over files
        for file in path.iterdir():
            raw = pd.read_table(file, decimal=decimal, index_col=0)
            raw.index.rename('wl', inplace=True)
            table_list.append(raw)

        # Concatenate and transpose
        curves = pd.concat(table_list, axis=1).transpose()

        # Format wavelengths to integer
        curves.columns = curves.columns.map(int)

        # Force unique names
        if any(curves.index.value_counts() > 1):
            curves.index = (curves.index + '_' +
                            curves.groupby(level=0).cumcount().map(str))


    # -------------------------------------------------------------------------
    # Further processing of data
    # -------------------------------------------------------------------------

    # Check for NaN
    if curves.isna().any().any():
        print('Warning: there are NaNs in the data')

    # Parse names
    parsed = curves.index.map(lambda x: parse(format_string, x).named)

    # Create dataframe with metadata
    meta = pd.DataFrame(parsed.tolist(), index=curves.index)

    # Create column to identify sample and reference measurements
    meta['is_ref'] = [ref_pattern in name for name in meta.index]
    meta['is_sample'] = [sample_pattern in name for name in meta.index]

    # Final dataframe
    df = meta.merge(curves, left_index=True, right_index=True)
    df.index.name = 'key'

    # Return data
    return df
