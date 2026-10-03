# %% Dependencies ----------------------------------------------------------------------

import prolab as pl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# %% Assessment of blanks from the PerkinElmer instrument ------------------------------

# Format string
blank_string = '{camp}_{pattern}_{id}_{rmode}_{msr}'

# Blanks from all measurements (performed by Carol)
rblanks = pl.read_files(
    path='data/TR_AQUARELA/all_blanks',
    instrument='perkinelmer',
    format_string=blank_string,
    blank_pattern='REF',
    sample_pattern='AMO',
    depig_pattern='EXT',
    decimal='.'
    )

# Blanks from the last dedicated blank measurements (performed by Carol)
rnblanks = pl.read_files(
    path='data/TR_BLANKS_AQUARELA',
    instrument='perkinelmer',
    format_string=blank_string,
    blank_pattern='REF',
    sample_pattern='AMO',
    depig_pattern='EXT',
    decimal='.'
    )

# Transform in Spectra objects
blanks = pl.Spectra(rblanks)
nblanks = pl.Spectra(rnblanks.loc[rnblanks.is_blank, :])


# %% Plot raw data ---------------------------------------------------------------------

rblanks_melt = rblanks.reset_index().melt(id_vars=rblanks.reset_index().columns[:10],
                                          var_name='wl',
                                          value_name='absorbance')
rblanks_melt['wl'] = rblanks_melt.wl.map(int)

rnblanks_melt = rnblanks.reset_index().melt(id_vars=rnblanks.reset_index().columns[:10],
                                           var_name='wl',
                                           value_name='absorbance')
rnblanks_melt['wl'] = rnblanks_melt.wl.map(int)

fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300, sharex=True, sharey=True)
sns.lineplot(data=rblanks_melt, x='wl', y='absorbance', units='key',
             estimator=None, hue='rmode', lw=.75, alpha=.5, ax=axs[0], legend=False)
sns.lineplot(data=rnblanks_melt, x='wl', y='absorbance', units='key',
             estimator=None, hue='rmode', lw=.75, alpha=.5, ax=axs[1])
sns.despine()

axs[0].set(ylabel='Absorbance (AU)')

for ax in axs:
    ax.grid(which='both', color='black', lw=.25)
    ax.set(xlabel='Wavelength (nm)')

axs[1].legend(title='Mode', edgecolor='w', ncols=2)
plt.show()


# %% Consistency check -----------------------------------------------------------------

# Consistency check of the blanks from all measurements
blanks.consistency.check(
    measurements='blank',
    groupby='rmode',
    std_threshold=0.0075,
    plot_suspicious=True,
    legend=False,
    remove_suspicious=True,
    recursive=True,
    inplace=True
    )

# Same, but from the dedicated blank measurements
nblanks.consistency.check(
    measurements='blank',
    groupby='rmode',
    std_threshold=0.0075,
    plot_suspicious=True,
    legend=False,
    remove_suspicious=True,
    recursive=True,
    inplace=True
    )

# plot the blanks from all measurements
fig, ax = plt.subplots(figsize=(5.75, 5.75 / 2), dpi=300)
blanks.raw_spectra.transpose().plot(legend=False, lw=.5, color='blue', ax=ax)
nblanks.raw_spectra.transpose().plot(legend=False, lw=.5, color='red', ax=ax)
plt.grid(which='major', color='gray', lw=.5)
ax.set(xlabel='Wavelength (nm)')
ax.set(ylabel='Absorbance (AU)')
sns.despine(ax=ax)
plt.show()


# %% Correction of breaks --------------------------------------------------------------

# Correct breaks semiautomatically (user indicated)
# blanks_cor = blanks.correction.treat_breaks(inline=True)

# Save corrected data
# blanks_cor.raw_data.to_csv('data/TR_AQUARELA/all_blanks_filtered_corrected.csv',
#                            index=True, decimal='.', sep=';')

# %% Upload corrected data -------------------------------------------------------------

blanks_cor = pd.read_csv('data/TR_AQUARELA/all_blanks_filtered_corrected.csv',
                         index_col=0, decimal='.', sep=';')
blanks_cor = pl.Spectra(blanks_cor)

# %% Mean blanks -----------------------------------------------------------------------

# Create and save median blank curves for reference
mean_blanks = (blanks_cor
               .raw_data
               .set_index('rmode')
               .filter(regex=r'\d')
               .groupby(level=0)
               .mean())
mean_blanks.columns = blanks_cor.wls

# Save mean blanks
mean_blanks.to_csv('data/TR_AQUARELA/standard_blanks.csv',
                   index=True, decimal='.', sep=';')

# Plot them
fig, ax = plt.subplots(figsize=(5.75, 5.75 / 2), dpi=300)
blanks_cor.raw_spectra.transpose().plot(legend=False, lw=.5, color='gray', ax=ax)
mean_blanks.transpose().plot(legend=False, lw=1, color='black', ax=ax)
plt.grid(which='major', color='gray', lw=.5)
ax.set(xlabel='Wavelength (nm)')
ax.set(ylabel='Absorbance (AU)')
sns.despine(ax=ax)
plt.show()
