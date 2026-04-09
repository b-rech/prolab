# -*- coding: utf-8 -*-
'''
Script: io.py
Description: functions to read and write data.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python version: 3.11

Dependencies:
    - matplotlib
    - numpy
    - pandas
    - scipy

Usage:
    python example.py
'''

# %% Dependencies

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.optimize


# %% Class definition

class Spectra:
    '''
        Class to handle laboratory spectral measurements.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataframe containing spectral measurements and metadata.
            Spectral wavelengths must be provided as column names
            (e.g., 450, 451, ...).

        id_attr : str
            The name of the column in `data` that will be used to identify the
            different measurements, e.g., station ID or salinity value.

        station_pattern : str
            A string used to filter sample measurements (e.g., `'point'`).

        reference_pattern : str
            A string used to filter reference measurements (e.g., `'milliq'`).
            Use `None` if no reference measurements are provided.

        Attributes
        ----------
        raw_data : pandas.DataFrame
            Original input dataframe.
        acdom : pandas.DataFrame
            Dataframe containing only spectral absorption values.
        meta : pandas.DataFrame
            Dataframe containing metadata columns.
        wavelengths : numpy.ndarray
            Array of spectral wavelengths.
        fit_df : pandas.DataFrame
            Results of spectral fits (created after calling `expfit`).

        Examples
        --------
        >>> from prolab import Spectra
        >>> spec = Spectra(data)
        >>> spec.expfit()
        >>> spec.fit_df

        Notes
        -----
        Spectral wavelengths are automatically detected from numeric
        column names.
        '''

    # -------------------------------------------------------------------------
    # Initializing function
    # -------------------------------------------------------------------------
    def __init__(self, data: 'pd.DataFrame'):
        """
        Initialize a `Spectra` object.

        Parameters
        ----------
        data : pd.DataFrame
            A Pandas DataFrame resulting from the
            `prolab.io.read_files` function.
        """

        # Set atrributes
        # --------------

        # Raw data
        self.raw_data = data

        # Absorptance curves
        self.absorptance = self.raw_data.filter(regex='\d')

        # Wavelengths
        self.wls = self.absorptance.columns.map(int).to_numpy()

        # Update columns to numeric
        self.absorptance.columns = self.wls

        # Columns with metadata
        meta_cols = data.columns.difference(self.absorptance.columns)

        # Create attributes from metadata
        for col in meta_cols:
            setattr(self, col, data[col])


    # -------------------------------------------------------------------------
    # Representation strings
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'Spectra(observations={len(self.absorptance)}, ' +
                f'wavelengths={len(self.wls)})')
    def __str__(self):
        return (f'Object Spectra with {len(self.absorptance)} observations and ' +
                f'{len(self.wls)} wavelengths')


    # -------------------------------------------------------------------------
    # Method: consistency analysis
    # -------------------------------------------------------------------------
    def consistency(self, std_threshold=.002, include_ref=False,
                    remove_suspicious=False, plot_suspicious=False,
                    plot_path=None):
        """
        Performs a consistency analysis of the curves.

        Parameters
        ----------
        std_threshold : float, optional
            Multiple measurements of the same station that have a standard
            deviation greater than this values will be marked as suspicious.
            The default is 0.1.
        include_ref : bool, optional
            Whether the reference measurements should be included in the
            analysis. The default is False.
        remove_suspicious : bool, optional
            Whether the suspicious curves should be removed; if True, a new
            Spectra object is returned. The default is False.
        plot_suspicious : bool, optional
            Whether to plot the suspicious measurements. The default is False.
        plot_path : str, optinal
            If provided, the plots are saved at the path. The default is None.

        Notes
        -----
        When `remove_suspicious=True`, each suspicious station' replicate is
        compared with the median curve, and the replicate with the highest
        absolute difference is removed.It means that only one of the curves is
        removed.

        It is recommended to run the consistency analysis again after the
        removal to ensure that the bad measurements have been effectively
        disregarded.

        Returns
        -------
        prolab.Spectra
            A new `Spectra` object is return if suspicious curves are removed.
            Otherwise, the same object is return.
        """

        # Include reference measurements in the analysis...
        if include_ref:
            data = self.raw_data
            absorptance = self.absorptance

        # ...or analyze only sample curves
        else:
            data = self.raw_data.loc[~self.is_ref]
            absorptance = self.absorptance.loc[~self.is_ref]

        # Spectral standard deviation
        std = data.set_index('id').filter(regex='\d').groupby(level=0).std()

        # Std over threshold
        std_over = (std > std_threshold).sum(axis=1) > 0

        # IDs with std over threshold
        ids_over = std_over.loc[std_over].index.tolist()

        # If suspicious stations exist
        if len(ids_over) > 0:

            print(f'IDs with std > {std_threshold}:')

            for s in ids_over:

                print(s)

                # Get suspicious curves
                suspicious_curves = data.loc[data.id == s].filter(regex='\d')

                # Plot them if required
                if plot_suspicious:
                    fig, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
                    suspicious_curves.transpose().plot(legend=True,
                                                       lw=0.5, ax=ax)
                    ax.set(xlabel='Wavelength (nm)', ylabel='Absorptance (AU)',
                           title=f'Suspicious curve: ID {s}')
                    ax.grid(which='both', lw=.25, color='black')
                    ax.legend(edgecolor='black', fancybox=False)
                    sns.despine()
                    if plot_path:
                        plt.savefig(plot_path / f'suspicious_id_{s}.jpg')
                    plt.show()

            # Calculate the median curves of each ID
            median = (
                data.groupby('id')
                .transform('median', numeric_only=True)
                .filter(regex='\d')
                )
            median.columns = median.columns.map(int)

            # Calculate mean absolute deviation (MAD) from the median curve
            mad = pd.DataFrame(absorptance.sub(median).abs()\
                               .sum(axis=1).rename('mad'))

            # Add IDs
            mad = mad.join(self.id)

            # Keys with highest MAD
            mad_max = mad.groupby('id').idxmax()

            # Intersection of keys with higher MAD and IDs that are suspicious
            self.suspicious = mad_max.loc[ids_over].mad.tolist()

            # Remove suspicious data
            if remove_suspicious:
                to_keep = data.index.difference(self.suspicious).tolist()
                filtered = data.filter(items=to_keep, axis=0)
                print('Suspicious curves removed')
                return Spectra(filtered)
            else:
                return self

        # If suspicious stations do not exist
        else:
            self.suspicious = None
            print('No suspicious curves')
            return self


    # -------------------------------------------------------------------------
    # Method: retrieve absorption
    # -------------------------------------------------------------------------
    def get_absortion(self, reference=None, wl_to_offset=600, pathlength=.1):
        """
        Retrieve absorption from absorptance curves.

        Parameters
        ----------
        reference : pd.DataFrame, optional
            If `None`, the reference measurements within the data will be used.
            If a single curve is provided (wavelengths in the columns),
            the same curve will be used to correct all measurements.
            If multiple curves are provided (salinity curves), the algorithm
            will select that closer to the sample at 685 nm, following the
            IOCCG protocol. The default is None.
        wl_to_offset : int, optional
            The absorption at this wavelength will be subtracted from the whole
            absorption spectrum. The default is 600.
        pathlength : float, optional
            The instrument optical path length in meters. The default is .1.

        Returns
        -------
        prolab.Spectra
            The same object is returned.
            A new attribute `absorption` is added inplace.
        """

        # Get mean curves
        mean = (
            self.raw_data.loc[~self.is_ref]
            .groupby('id')
            .mean(numeric_only=True)
            .filter(regex='\d')
            )
        mean.columns = mean.columns.map(int)

        # In case a reference is provided
        if reference is not None:
            blank = reference
            blank.columns = blank.columns.map(int)
        # Otherwise, use those from the data
        else:
            blank = (self.absorptance.loc[self.is_ref]
                     .median()
                     .rename('ref')
                     .to_frame()
                     .transpose()
                     )

        # Min and max wavelengths available
        wlmin = max(self.wls.min(), blank.columns.min())
        wlmax = min(self.wls.max(), blank.columns.max())
        wl_range = range(wlmin, wlmax + 1)

        # Filter wavelengths
        self.mean_absorptance = mean[wl_range]
        blank = blank[wl_range]

        # For single blank curves
        if len(blank) == 1:
            self.absorptance_corr = self.mean_absorptance.sub(blank.squeeze(),
                                                              axis=1)

        # For multiple blank curves
        else:
            # Dictionary to receive corrected curves
            corr_dict = {}

            # Dictionaries to save reference curves and their names
            ref_dict = {}
            ref_sal_dict = {}

            # Iterate over curves
            for st, curve in self.mean_absorptance.iterrows():

                # Select a reference curve
                ref_sel = (
                    abs(blank.loc[:, 685:].sub(curve.loc[685:], axis=1))
                    .mean(axis=1)
                    .sort_values()
                    .index[0]
                    )

                # Subtract selected reference
                corr_dict[st] = (curve - blank.loc[ref_sel])

                # Save the selected reference curve
                ref_dict[st] = blank.loc[ref_sel]
                ref_sal_dict[st] = ref_sel

            # Corrected data
            self.absorptance_corr = pd.DataFrame(corr_dict).transpose()

            # Reference used
            self.ref_curves = pd.DataFrame(ref_dict).transpose()
            self.ref_salinity = pd.DataFrame(ref_sal_dict, index=[0])

        # Apply offset
        absorptance_off = self.absorptance_corr.sub(
            self.absorptance_corr[wl_to_offset], axis=0
            )

        # Calculate absorption
        self.absorption = 2.3 * absorptance_off / pathlength

        return self


    # -------------------------------------------------------------------------
    # Method: fit exponential curve
    # -------------------------------------------------------------------------
    def expfit(self, wl_ref=443, wl_range=(350, 500)):
        """
        Fits exponential curves.

        Parameters
        ----------
        wl_ref : int, optional
            Wavelength of reference for the fitting. The default is 443.
        wl_range : tuple, optional
            Limits of the wavelength range to be used in the curve fitting.
            The default is (350, 500).

        Returns
        -------
        prolab.Spectra
            The same object is returned.
            A new attribute `fitted` is added inplace.
        """
        # Exponential curve to fit
        def exp_curve(wl, abs_ref, slope):
            return abs_ref * np.exp(-slope * (wl - wl_ref))

        # Initial guesses for abs_ref and slope
        init_guess = (0.1, 0.02)

        # Linearlize parameters
        ydata = self.absorption.loc[:, range(wl_range[0], wl_range[1])]
        x= ydata.columns.map(int).to_numpy()

        # Dictionary to store data
        fit_dict = {'id': [], f'acdom{wl_ref:.0f}': [], 'slope': []}

        # Fit coefficients
        for st, y in ydata.iterrows():

            # Fit curve
            coeffs, _ = scipy.optimize.curve_fit(exp_curve, x, y, init_guess)

            # Store data
            fit_dict['id'].append(st)
            fit_dict[f'acdom{wl_ref:.0f}'].append(coeffs[0])
            fit_dict['slope'].append(coeffs[1])

        # Final dataframe
        self.fitted = pd.DataFrame(fit_dict).set_index('id')

        return self
