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
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Special modules
import prolab as pl


# %% Salinity

# Format of the salinity filenames
format_sal = '{pattern:3}{unit:4}_{id:d}{replicate}_{campaign}'

# Open curves
raw_sal = pl.read_files(path='data/sal_curves_10cm',
                        sample_pattern='sal',
                        ref_pattern='None',
                        format_string=format_sal)

# Plot raw curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
raw_sal.filter(regex='^\d').transpose().plot(legend=False, ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance')
plt.show()

# Transform in Spectra object
sal_spec = pl.Spectra(raw_sal)

# Consistency check
sal_spec.consistency(std_threshold=.002)
# Remove suspicious
sal_spec = sal_spec.consistency(std_threshold=.002, remove_suspicious=True)
sal_spec.consistency(std_threshold=.002)

# Mean absorptance
mean_abspt = (sal_spec
              .raw_data
              .groupby('id')
              .mean(numeric_only=True)
              .filter(regex='\d')
              )
mean_abspt.index = [x/100 for x in mean_abspt.index]
mean_abspt.columns = mean_abspt.columns.map(int)

# Plot mean original curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
mean_abspt.transpose().plot(legend=True, ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance')
ax.legend(fontsize=5)
plt.show()

# Plot absorptance at 685 nm versus salinity
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
sns.scatterplot(mean_abspt[685], ax=ax)
sns.lineplot(mean_abspt[685], ax=ax, alpha=.5)
ax.set(xlabel='Salinity', ylabel='Absorptance (AU)')
plt.show()

# Prepare index
new_index = pd.Index(np.arange(0, 41, 0.1, dtype=float))
comb_index = (new_index
              .union(mean_abspt.index)
              .sort_values()
              .round(3)
              .drop_duplicates()
              )

# Iterpolate curves
final_abs = mean_abspt.reindex(comb_index).interpolate(method='index')
final_abs.index.name = 'salinity'

# Plot interpolated curves
_, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
final_abs.transpose().plot(legend=False, ax=ax, cmap='turbo', lw=.5)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorption $\mathrm{m^{-1}}$')
plt.show()

# Save curves
final_abs.to_csv('data/sal_curve_interpolated_202604.csv', sep=';')
