# %%
import prolab as pl

# %% Assessment of blanks from the PerkinElmer instrument

raw_blanks = pl.read_files('data/TR_AQUARELA/all_blanks',
                           instrument='perkinelmer',
                           format_string='{camp}_{pattern}_{id}_{rmode}_{msr}',
                           blank_pattern='REF',
                           sample_pattern='AMO',
                           depig_pattern='EXT',
                           decimal='.')

spec_blanks = pl.Spectra(raw_blanks)

spec_blanks.consistency.check(measurements='blank',
                              groupby='rmode',
                              std_threshold=0.01,
                              plot_suspicious=True,
                              legend=False,
                              remove_suspicious=True,
                              recursive=True,
                              inplace=True)

spec_blanks.raw_spectra.transpose().plot(legend=False, lw=.5)
plt.show(block=True)

# %%

raw = pl.read_files('data/TR_20260504_csv',
                    instrument='perkinelmer',
                    format_string='{camp}_{pattern}_{id}_{rmode}_{msr}',
                    blank_pattern='REF',
                    sample_pattern='AMO',
                    depig_pattern='EXT',
                    decimal=',')

raw = pl.read_files('data/shimadzu_tr',
                    instrument='shimadzu',
                    format_string='{camp}_{pattern}_{rmode}_{id}_{msr}',
                    blank_pattern='Branco',
                    sample_pattern='Amostra',
                    depig_pattern='ext',
                    decimal='.')


spec = pl.Spectra(raw)

test = spec.correction.treat_breaks(inline=False)


spec.consistency.check(measurements='blank',
                       groupby=('rmode'),
                       std_threshold=0.001,
                       plot_suspicious=True,
                       remove_suspicious=False,
                       recursive=True,
                       legend=False,
                       inplace=False)

spec.particulate.get_absorption(vol_diameter=(1, 1), type_measurement='absorbance',
                                percentual=True, plot_raw=True, use_tau=False,
                                wl_range=(400, 800))

spec.part_abs_depig.transpose().plot()

spec.part_abs_total.sub(spec.part_abs_depig).transpose().plot()

