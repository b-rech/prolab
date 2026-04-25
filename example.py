# -*- coding: utf-8 -*-
"""
Script: example.py
Description: Example of usage for sample processing.
Author: Bruno Rech
Institution: INPE/CEBIMar
Created: 2026-03-12
"""

# %% Dependencies

# Libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

# Special modules
import prolab as pl

# Path to data
data_path = Path('data/WPI/05mar2026')
save_path = Path('data/processed')
save_path.mkdir(exist_ok=True, parents=True)
plot_path = save_path / 'plots'
plot_path.mkdir(exist_ok=True, parents=True)

# Save data?
SAVE_FIGURES = False
SAVE_DATA =    False


# %% Open data

# Absorptance data
raw_data = pl.read_files(path=data_path,
                         instrument='wpi',
                         sample_pattern='ponto',
                         ref_pattern='milliq',
                         format_string='{pattern}{id:d}{rep}_{camp}')

# Plot raw curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
raw_data[~raw_data.is_ref]\
    .set_index('id')\
    .filter(regex='^\d')\
    .transpose()\
    .plot(legend=False, lw=.3, color='black', ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
       title='All measurements (with replicates)')
ax.grid(which='both', lw=.25, color='black')
sns.despine()
if SAVE_FIGURES:
    plt.savefig(plot_path / f'{data_path.stem}_all_measurements.png')
plt.show()

# Salinity curve to use as reference
salinity = pd.read_csv('data/sal_curve_interpolated_202604.csv',
                       sep=';', index_col=0)

# Create object
spectra = pl.Spectra(data=raw_data)

# Check consistency
spectra.consistency(std_threshold=.005, plot_suspicious=True)


# %% Remove suspicious

# In case you want to remove suspicious curve, run the consistency analysis
# again with remove_suspicious=True
spectra = spectra.consistency(std_threshold=.005, remove_suspicious=True)
spectra.consistency(std_threshold=.005, plot_suspicious=True)


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
