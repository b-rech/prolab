# %%
import prolab as pl

format_string = '{camp}_{pattern}_{id}_{rep}_{refilter}_{parameter}'

raw = pl.read_files('data/cdom_shimadzu',
                    instrument='shimadzu',
                    format_string=format_string,
                    sample_pattern='SAMPLE',
                    blank_pattern='BLANK',
                    decimal=',')

# %% Processing CDOM absorption data

# Create Spectra object from raw data
spectra = pl.Spectra(raw)

# Verify consistency of measurements
spectra.consistency.check(std_threshold=0.005, groupby=['id', 'refilter'],
                          plot_suspicious=True, measurements='sample')

# Calculate absorption and fit exponential model
spectra.cdom.get_absorption(mean_replicates=False,
                            wl_null_point_correction=None)
spectra.cdom.expfit(wl_ref=440, wl_range=(350, 700))

# Get data
absorption = spectra.absorption
fitted = spectra.cdom_fitted


absorption['id'] = spectra.id
absorption['refilter'] = spectra.refilter


final_melt = absorption.melt(id_vars=['id', 'refilter'],
                             var_name='wavelength',
                             value_name='absorption')

final_melt = final_melt.loc[final_melt['wavelength'].between(400, 600)]

# %%

import seaborn as sns
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

sns.lineplot(data=final_melt,
             x='wavelength',
             y='absorption',
             style='refilter',
             hue='id',
             estimator='mean',
             lw=.5,
             ax=ax,
             palette=['red', 'blue', 'green'])
plt.grid(which='both', axis='both', color='lightgrey', alpha=.5)
plt.yscale('log')
sns.despine()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption (m$^{-1}$)')

# %%
plt.savefig(r'C:\Users\brech\Desktop\repositories\prolab\cdom_shimadzu_absorption.png',
            dpi=300)

# %%
