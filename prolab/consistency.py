'''
Module: consistency.py
Purpose: Replicate consistency analysis and suspicious-curve filtering.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Updated: 2026-09-27
Python: 3.11+

Dependencies: pandas, matplotlib, seaborn
Public API: ConsistencyProcessor.check
'''

from math import isfinite
from numbers import Real

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class ConsistencyProcessor:
    '''
    Consistency analysis associated with a Spectra instance.
    '''

    def __init__(self, spectra):
        '''
        Create a consistency processor attached to a Spectra object.
        '''
        self.spectra = spectra

    def check(self,
            std_threshold=.1,
            measurements='all',
            groupby='id',
            remove_suspicious=False,
            plot_suspicious=False,
            recursive=False,
            legend=True,
            plot_path=None,
            inplace=False):
        '''
        Check replicate spread and optionally remove suspicious curves.

        Parameters
        ----------
        std_threshold : float, optional
            Maximum accepted standard deviation at each wavelength.
        measurements : {'all', 'sample', 'blank'}, optional
            Subset of measurements to analyze.
        groupby : str or list, optional
            Metadata column(s) defining a station or measurement group.
        remove_suspicious : bool, optional
            Remove the highest-deviation curve from each suspicious group.
        plot_suspicious : bool, optional
            Display plots for suspicious groups.
        recursive : bool, optional
            Repeat removal until no suspicious groups remain.
        legend : bool, optional
            Show a legend on suspicious-curve plots.
        plot_path : pathlib.Path, optional
            Directory where suspicious-curve plots are saved.
        inplace : bool, optional
            When ``True``, remove suspicious curves from the associated
            :class:`Spectra` object and return that same object. When
            ``False``, return a filtered replacement instead.

        Returns
        -------
        Spectra
            The associated object, or a filtered replacement when removal is
            requested.
        '''

        spectra = self.spectra

        # Fail early so invalid settings do not produce misleading results.
        if (not isinstance(std_threshold, Real)
            or not isfinite(std_threshold)):
            raise ValueError('std_threshold must be a finite non-negative number')
        if std_threshold < 0:
            raise ValueError('std_threshold must be a finite non-negative number')
        if measurements not in {'all', 'sample', 'blank'}:
            raise ValueError("measurements must be 'all', 'sample', or 'blank'")

        # Normalize one grouping column and multiple grouping columns to the
        # same list form. Pandas returns scalar labels for one key and tuples
        # for multiple keys; the rest of this method handles both forms.
        group_keys = [groupby] if isinstance(groupby, str) else list(groupby)
        if not group_keys:
            raise ValueError('groupby must contain at least one column')
        missing = set(group_keys).difference(spectra.raw_data.columns)
        if missing:
            raise KeyError(f'groupby columns not found: {sorted(missing)}')

        # Use Spectra's canonical wavelength list instead of selecting columns
        # with a regex, which could accidentally include metadata columns.
        wavelength_columns = [wl for wl in spectra.wls
                              if wl in spectra.raw_data.columns]
        if not wavelength_columns:
            raise ValueError('No wavelength columns are available for analysis')

        # Restrict the analysis to samples, blanks, or the complete dataset.
        # The filtered frame keeps the original row index so suspicious rows
        # can later be removed from the complete raw dataset.
        if measurements == 'sample':
            data = spectra.raw_data.loc[spectra.is_sample].copy()
        elif measurements == 'blank':
            data = spectra.raw_data.loc[spectra.is_blank].copy()
        else:
            data = spectra.raw_data.copy()

        # A group is suspicious when the replicate standard deviation exceeds
        # the threshold at least at one wavelength. The sample standard
        # deviation naturally leaves groups with one replicate non-suspicious.
        grouped = data.groupby(group_keys, sort=False, dropna=False)
        suspicious_groups = []
        for label, group_data in grouped:
            spread = group_data[wavelength_columns].std()
            if spread.gt(std_threshold).any():
                suspicious_groups.append((label, group_data))

        if suspicious_groups:
            # Plot only groups that failed the spread check. Plotting is kept
            # inside this branch so normal analyses do not create figures.
            for label, group_data in suspicious_groups:
                if plot_suspicious:
                    figure, axis = plt.subplots(figsize=(4, 3), dpi=300)
                    group_data[wavelength_columns].transpose().plot(
                        legend=legend, lw=.5, ax=axis)
                    label_text = label if len(group_keys) > 1 else str(label)
                    axis.set(xlabel='Wavelength (nm)', ylabel='Measurement',
                             title=f'Suspicious curve: ID {label_text}')
                    axis.grid(which='both', lw=.25, color='black')
                    if legend:
                        axis.legend(frameon=False)
                    sns.despine()
                    if plot_path:
                        figure.savefig(plot_path / f'suspicious_id_{label}.jpg')
                    plt.show()
                    plt.close(figure)

            # Compare every curve with its group median. The resulting score
            # is the sum of absolute deviations across wavelengths; the row
            # with the largest score is treated as the suspicious replicate.
            median = grouped[wavelength_columns].transform('median')
            mad = (data[wavelength_columns].sub(median).abs().sum(axis=1).rename('mad'))
            spectra.suspicious = [
                mad.loc[group_data.index].idxmax()
                for _, group_data in suspicious_groups
            ]

            # Optionally remove the worst replicate from the original dataset.
            # A recursive run is useful when a second-outlier group becomes
            # visible only after the first suspicious curve is removed.
            if remove_suspicious:
                to_keep = spectra.raw_data.index.difference(spectra.suspicious)
                filtered = spectra.raw_data.loc[to_keep]
                if inplace:
                    # Drop rows on the original table, then rebuild the
                    # derived spectra, metadata, and processor attributes.
                    removed = list(spectra.suspicious)
                    spectra.raw_data.drop(index=removed,
                                          inplace=True)
                    spectra.__init__(spectra.raw_data)
                    spectra.suspicious = removed
                    target = spectra
                else:
                    target = type(spectra)(filtered)
                if recursive:
                    return target.consistency.check(
                        std_threshold, measurements, groupby,
                        remove_suspicious, plot_suspicious, recursive,
                        legend, plot_path, inplace)
                return target
            return spectra

        spectra.suspicious = None
        return spectra
