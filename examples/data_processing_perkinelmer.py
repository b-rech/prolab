'''
Script: data_processing_perkinelmer.py
Purpose: Process PerkinElmer transmittance-reflectance measurements,
         correct spectral breaks, and save the corrected dataset.
Author: Bruno Rech
Institution: INPE
Created: 2026-10-03
Python: 3.11+

Dependencies: prolab, matplotlib, seaborn, pandas
'''

# %% Dependencies ----------------------------------------------------------------------

import prolab as pl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


# %% Upload data -----------------------------------------------------------------------

# For PerkinElmer measurements, the data must be in a general folder with a subfolder
# named "raw" containing the raw CSV files.
# In the general folder also must contain the CSV file linking the statiod IDs with
# volume and clearance diameter values.

# Path to the data folder
data_path = Path('data/TR_20260504')

# Open standard blanks for verifications
standard_blanks = pd.read_csv('data/standard_blanks_perkinelmer.csv',
                              sep=';', decimal='.', index_col=0)
standard_blanks.columns = standard_blanks.columns.map(int)

# Open volume and diameter information
vol_diameter = pd.read_csv(data_path / 'vol_diameter.csv',
                           sep=';', decimal='.', index_col=0)
vol_diameter['volume'] /= 1e6    # mL to m³
vol_diameter['diameter'] /= 1000  # mm to m

# Load dataset
raw = pl.read_files(
    path=data_path / 'raw',
    instrument='perkinelmer',
    format_string='{camp}_{pattern}_{id}_{rmode}_{msr}',
    blank_pattern='REF',
    sample_pattern='AMO',
    depig_pattern='EXT',
    decimal='.'
    )

# Create a Spectra object
spec = pl.Spectra(raw)


# %% Correction of breaks --------------------------------------------------------------

# Run correction
spec_cor = spec.correction.treat_breaks(inline=False)

# Save corrected spectra
spec_cor.raw_data.to_csv(data_path / 'break_corrected.csv',
                         index=True, decimal='.', sep=';')


# %% Check consistency of blanks -------------------------------------------------------

# Get only blanks
blanks = spec_cor.raw_spectra.loc[spec_cor.is_blank]
blanks.insert(0, 'rmode', spec_cor.rmode.loc[spec_cor.is_blank])

# Get mean blanks
mean_blanks = blanks.groupby('rmode').mean()
mean_blanks.columns = mean_blanks.columns.map(int)

# Compare blanks with the standard blank curves
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
mean_blanks.transpose().plot(legend=False, color='red', ax=ax)
standard_blanks.transpose().plot(legend=False, color='black', ax=ax)
ax.set(xlabel='Wavelength (nm)', ylabel='Absorbance (AU)')
ax.grid(which='both', lw=.25, color='black')
sns.despine()
plt.show()


# %% Calculate particulate absorption --------------------------------------------------

# Retrieve curves
spec_cor.particulate.get_absorption(
    vol_diameter=vol_diameter,
    percentual=False,
    type_measurement='absorbance',
    wl_offset=800,
    use_tau=True,
    trans_pattern='T',
    wl_range=None,
    plot_raw=True,
    plot_absorbance=False,
    split_total_depig_blank=False,
    custom_blanks=None
)

# Save particulate absorption curves
spec_cor.part_abs_total.to_csv(data_path / 'particulate_abs_total.csv',
                              index=True, decimal='.', sep=';')
spec_cor.part_abs_depig.to_csv(data_path / 'particulate_abs_depig.csv',
                               index=True, decimal='.', sep=';')
spec_cor.part_abs_total.subtract(spec_cor.part_abs_depig).to_csv(
    data_path / 'particulate_abs_phyto.csv',
    index=True, decimal='.', sep=';'
    )


# %% Plot particulate absorption curves ------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=300, sharey=True)
spec_cor.part_abs_total.transpose().plot(ax=axes[0], legend=False, lw=.5)
spec_cor.part_abs_depig.transpose().plot(ax=axes[1], lw=.5)
spec_cor.part_abs_total.subtract(spec_cor.part_abs_depig).transpose().plot(ax=axes[2],
                                                                           lw=.5,
                                                                           legend=False)
axes[1].legend(frameon=False, fontsize=6, title='Station')
sns.despine()

for ax, title in zip(axes, ['Total', 'Detritus', 'Phytoplankton']):
    ax.set(xlabel='Wavelength (nm)', ylabel=r'Absorption $\mathrm{(m^{-1})}$')
    ax.set_title(title)

plt.show()
