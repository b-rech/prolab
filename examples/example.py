# -*- coding: utf-8 -*-
"""
Script: example.py
Description: Example of usage for sample processing.
Author: Bruno Rech
Institution: INPE/CEBIMar
Created: 2026-03-12
"""

# %% Dependencies -------------------------------------------------------------

# Libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Special modules
import prolab as pl

# Path to data
data_path = Path('data/TR_20260506')
save_path = Path('data/processed')
save_path.mkdir(exist_ok=True, parents=True)
plot_path = save_path / 'plots'
plot_path.mkdir(exist_ok=True, parents=True)

# Save data?
SAVE_FIGURES = False
SAVE_DATA =    False


# %% Create a master blank ----------------------------------------------------

# Folder with all blanks
path_blanks = Path('data/blanks_perkinelmer')

# Read blanks
raw_blanks = pl.read_files(
    path=path_blanks,
    instrument='perkin-elmer',
    sample_pattern='AMO',
    blank_pattern='REF',
    trans_pattern='T',
    depig_pattern='EXT',
    format_string='{camp}_{pattern}_{id}_{rmode}_{rtype}'
    )

raw_blanks = raw_blanks.filter(regex='^\d')
raw_blanks_n = raw_blanks.div(raw_blanks.mean(axis=1), axis=0)

ddx = abs(np.gradient(raw_blanks_n.filter(regex='^\d').to_numpy(), axis=1))

ddx = pd.DataFrame(ddx, columns=raw_blanks.filter(regex='^\d').columns)

ddx_melted = ddx.melt()

ddx.T.plot(legend=False)

# Plot raw curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
raw_blanks\
    .transpose()\
    .plot(legend=False, lw=.3, color='black', ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
       title='All measurements (with replicates)')
ax.grid(which='both', lw=.25, color='black')
sns.despine()



# %% Break correction ---------------------------------------------------------

# Breaks in the spectra
breaks = [422, 558, 685, 795]

brange = np.arange(breaks[0]-4, breaks[0]+5, 1)

for i in raw_blanks.index:
    deviation = raw_blanks_n.loc[i][brange].std()
    if deviation > 0.005:
        #sns.lineplot(raw_blanks.loc[i], lw=1, color='black')

        # Select data
        upper = raw_blanks.loc[i][brange][1:]
        lower = raw_blanks.loc[i][brange][:-1]

        # Calculate difference
        diff = abs(upper.to_numpy() - lower.to_numpy())
        diff_idx = np.argmax(diff)
        upper_wl = upper.index[diff_idx]
        lower_wl = lower.index[diff_idx]

        corrected = raw_blanks.loc[i].copy()
        corrected = corrected.where(corrected.index < upper_wl, corrected + diff[diff_idx])

        sns.lineplot(corrected, lw=1, color='black')
        plt.xlim((415, 430))


# %% Open data ----------------------------------------------------------------

# Absorptance data
raw_data = pl.read_files(
    path=data_path,
    instrument='perkin-elmer',
    sample_pattern='AMO',
    blank_pattern='REF',
    trans_pattern='T',
    depig_pattern='EXT',
    format_string='{camp}_{pattern}_{id:d}_{rmode}_{rtype}'
    )

# Table with filtered volumes
volume_data = pd.read_csv('data/TR_VOLUMES_20260506.csv', sep=';')

# Create spectra object
spectra = pl.Spectra(raw_data)

from scipy.signal import savgol_filter

for st in spectra.curves.index:

    _, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
    ax2 = ax.twinx()

    test = pd.DataFrame(spectra.curves.loc[st])
    test = test.div(test.mean())

    ddx2 = pd.DataFrame(np.gradient(np.gradient(np.gradient(test.T.to_numpy()[0]))))
    ddx2 = pd.DataFrame(ddx2.div(ddx2.abs().sum())).abs()
    ddx2.set_index(test.index, inplace=True)
    ddx2.plot(ax=ax, color='r', lw=.3)

    test.plot(ax=ax2, color='black', lw=1)
    plt.xlim((415, 425))
    plt.show()


# %% Plot all curves together -------------------------------------------------

# Plot raw curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
raw_data[~raw_data.is_blank]\
    .set_index('id')\
    .filter(regex='^\d')\
    .transpose()\
    .plot(legend=False, lw=.3, color='black', ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
       title='All measurements (with replicates)')
ax.grid(which='both', lw=.25, color='black')
sns.despine()

# Save if required
if SAVE_FIGURES:
    plt.savefig(plot_path / f'{data_path.stem}_all_measurements.png')
plt.show()


# %% Plot blanks --------------------------------------------------------------

# Plot raw curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
raw_data[raw_data.is_blank]\
    .filter(regex='^\d')\
    .transpose()\
    .plot(legend=True, lw=1, ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
       title='All measurements (with replicates)')
ax.grid(which='both', lw=.25, color='black')
sns.despine()

# Save if required
if SAVE_FIGURES:
    plt.savefig(plot_path / f'{data_path.stem}_all_measurements.png')
plt.show()


# %% Consistency check --------------------------------------------------------

# Only possible when there are replicates.


# %% Get absorption

spectra.part_absorption(vol_diameter=None,
                        unit='absorbance',
                        wl_offset=800,
                        use_tau=True,
                        trans_pattern='T',
                        wl_range=None,
                        plot_raw=True,
                        plot_absorbance=True)



#%% Spectra correction

# Correct absorptance and retrieve absorption
spectra.get_absortion(pathlength=.1071, reference=salinity, wl_to_offset=600)

# Plot each station' absorption curve with the selected reference
for st, row in spectra.mean_absorptance.iterrows():
    _, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300,
                         constrained_layout=True)
    sns.lineplot(row, lw=1, color='blue', ax=ax, label='Raw')
    sns.lineplot(spectra.absorptance_corr.loc[st], lw=1, color='blue',
                 ax=ax, label='Corrected', ls='--')
    sns.lineplot(spectra.ref_curves.loc[st], lw=1, color='red',
                 ax=ax, label='Reference')
    ax.grid(which='both', lw=.25, color='black')
    ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
           title=f'ID {st}')
    sns.despine()
    if SAVE_FIGURES:
        plt.savefig(plot_path / f'correction_id_{st}.jpg')
    plt.show()

# Fit curve
spectra.expfit(wl_ref=443, wl_range=(350, 500))
spectra.fitted

# Plot fitted curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2),
                     dpi=300, constrained_layout=True)
spectra.absorption.filter(items=range(400, 601)).transpose().plot(legend=True,
                                                                  lw=1, ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel=r'Absorption $(\mathrm{m^{-1}})$')
ax.grid(which='both', lw=.25, color='black')
ax.legend(fontsize=7)
sns.despine()
if SAVE_FIGURES:
    plt.savefig(plot_path / f'absorption_spectra_{data_path.stem}.jpg')
plt.show()

# Save data
if SAVE_DATA:
    spectra.fitted.to_csv(save_path / f'{data_path.stem}_fitted_params.csv',
                          sep=';')


# %% Merge all files

# List to save tables
tables = []

# Iterate over files
for file in save_path.iterdir():
    if 'fitted_params' in file.stem:
        tables.append(pd.read_csv(file, sep=';', index_col=0))

# Create dataframe of CDOM data
cdom = pd.concat(tables, axis=0)
cdom.index = [f'P{x:02d}' for x in cdom.index]

# Open data
params = pd.read_csv('data/params_paranagua.csv', sep=';')
params.set_index('station', inplace=True)

# Join dataframes
df = params.merge(cdom, left_index=True, right_index=True, how='left')

if SAVE_DATA:
    df.to_csv(save_path / 'merged_data.csv', sep=';')
