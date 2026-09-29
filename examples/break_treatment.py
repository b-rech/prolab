# %% Dependencies

import prolab as pl
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Ensure that the QtAgg backend is used for interactive plotting
matplotlib.use("QtAgg")


# %% Open all sample measurements ------------------------------------------------------

# Open sample measurements
raw = pl.read_files('data/TR_AQUARELA/all_samples',
                    instrument='perkinelmer',
                    format_string='{camp}_{pattern}_{id}_{rmode}_{msr}',
                    blank_pattern='REF',
                    sample_pattern='AMO',
                    depig_pattern='EXT',
                    decimal='.')


# %% Break treatment -------------------------------------------------------------------

# Location (approximate) of the breaks
breaks_center = [795, 684, 558, 422]

# Range of wavelengths to check for breaks (in nm)
breaks_range = {f'{letter}': np.arange(wl + 1, wl - 2, -1)
                for letter, wl in zip(['A', 'B', 'C', 'D'], breaks_center)}

# Filter the raw data to only include numeric columns (wavelengths)
uncorrected = raw.filter(regex=r'\d')
uncorrected_norm = uncorrected.div(uncorrected.mean(axis=1), axis=0)

#%% Option I: correct all without checking for breaks ----------------------------------

#%matplotlib inline

for i in uncorrected.index[100:]:

    # Copy spectrum for correction
    spectrum_corr = uncorrected.loc[i].copy()

    fig, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
    sns.lineplot(uncorrected.loc[i], lw=.5, color='red', ax=ax)
    for wl_range, letter in zip(breaks_range.values(), breaks_range.keys()):
        ax.axvspan(wl_range[-1], wl_range[0], alpha=0.5, color='gray')
        ax.text(s=letter, x=wl_range[1], y=spectrum_corr.max(),
                color='black', fontsize=8)
    plt.show(block=True)

    test = uncorrected.loc[i, 774:786].copy()
    test.loc[776:784] = np.nan
    test.interpolate(method='linear', inplace=True)
    spectrum_corr.loc[test.index] = test

    to_correct = input('Indicate which break to correct (A, B, C, D, [None]): ')

    if to_correct:

        for id_range in to_correct:

            wl_range = breaks_range[id_range.upper()]
            wl_reference = wl_range[0]
            wl_break = wl_range[-1]

            offset = (2 * spectrum_corr.loc[wl_reference]
                    - spectrum_corr.loc[wl_break-2:wl_break].mean()
                    - spectrum_corr.loc[wl_reference:wl_reference+2].mean())

            spectrum_corr.loc[:wl_break] += offset

            # Interpolate the break
            middle = spectrum_corr.loc[wl_range].copy()
            middle.iloc[1:-1] = np.nan
            middle.interpolate(method='linear', inplace=True)
            spectrum_corr.loc[middle.index] = middle

        fig, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
        sns.lineplot(uncorrected.loc[i], lw=.5, color='red', ax=ax)
        sns.lineplot(spectrum_corr, lw=.5, color='green', ax=ax)
        for wl_range, letter in zip(breaks_range.values(), breaks_range.keys()):
            ax.axvspan(wl_range[-1], wl_range[0], alpha=0.5, color='gray')
            ax.text(s=letter, x=wl_range[1], y=spectrum_corr.max(), color='black', fontsize=8)
        plt.show(block=True)

# %%
