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
        rdata : pandas.DataFrame
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
        '''
        Initialize a `Spectra` object.

        Parameters
        ----------
        data : pd.DataFrame
            A Pandas DataFrame resulting from the
            `prolab.io.read_files` function.
        '''

        # ---------------
        # Set atrributes
        # ---------------

        # Raw data
        self.rdata = data

        # Spectral curves
        self.curves = self.rdata.filter(regex='\d')

        # Wavelengths
        self.wls = self.curves.columns.map(int).to_numpy()

        # Update columns to numeric
        self.curves.columns = self.wls

        # Columns with metadata
        meta_cols = data.columns.difference(self.curves.columns)

        # Create attributes from metadata
        for col in meta_cols:
            setattr(self, col, data[col])


    # -------------------------------------------------------------------------
    # Representation strings
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'Spectra(observations={len(self.curves)}, ' +
                f'wavelengths={len(self.wls)})')
    def __str__(self):
        return (f'Object Spectra with {len(self.curves)} observations and ' +
                f'{len(self.wls)} wavelengths')


    # -------------------------------------------------------------------------
    # Method: consistency analysis
    # -------------------------------------------------------------------------
    def consistency(self,
                    std_threshold=.1,
                    measurements='all',
                    groupby='id',
                    remove_suspicious=False,
                    plot_suspicious=False,
                    recursive=False,
                    legend=True,
                    plot_path=None):
        '''
        Performs a consistency analysis of the curves.

        Parameters
        ----------
        std_threshold : float, optional
            Multiple measurements of the same station that have a standard
            deviation greater than this value at any wavelength will be marked
            as suspicious. The default is 0.1.
        measurements : str, optional
            Use `all` (default) to analyze all curves together, `sample` to
            analyze only sample curves, or `ref` to analyze only
            reference curves.
        groupby : str or list, optional
            Indicate the attribute or list of attributes to be used for
            grouping the observations. The default is `id`.
        remove_suspicious : bool, optional
            Whether the suspicious curves should be removed; if `True`, a new
            Spectra object is returned. The default is `False`.
        plot_suspicious : bool, optional
            Whether to plot the groups with suspicious measurements.
            The default is `False`.
        recursive : bool, optional
            When `remove_suspicious=True`, will make the function run again
            until no suspicious measurements are left, or until a single
            measurement is left. The default is `False`.
        legend : bool, optional
            Whether to insert a legend when `plot_suspicious=True`.
            The default is `True`.
        plot_path : str, optinal
            If provided and if `plot_suspicious=True`, the plots are saved at
            this folder. The default is `None`.

        Notes
        -----
        When `remove_suspicious=True`, each suspicious station' replicate is
        compared with the median curve, and the replicate with the highest
        absolute difference is removed. It means that only one of the curves is
        removed at a time. The option `recursive=True` may be used, but make
        sure to use a reasonable threshold, otherwise the spectra will be
        dropped until only a single curve is left.

        Returns
        -------
        prolab.Spectra
        '''

        # Analyze only samples...
        if measurements == 'sample':
            data = self.rdata.loc[self.is_sample].copy()

        # ... or only references...
        elif measurements == 'ref':
            data = self.rdata.loc[self.is_ref].copy()

        # ... or everything together
        else:
            data = self.rdata.copy()

        # Spectral standard deviation
        std = (data
               .groupby(groupby)
               .std(numeric_only=True)
               .filter(regex='\d'))

        # Check if any wavelength has std over the threshold
        std_over = (std > std_threshold).sum(axis=1) > 0

        # IDs with std over threshold
        ids_over = std_over.loc[std_over].index.tolist()

        # If suspicious stations exist
        if len(ids_over) > 0:

            # Iterate over curves
            for s in ids_over:

                # Get suspicious curves
                suspicious_curves = (data
                                     .set_index(groupby)
                                     .sort_index()
                                     .loc[s]
                                     .filter(regex='\d'))

                # Plot them if required
                if plot_suspicious:
                    fig, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
                    suspicious_curves.transpose().plot(legend=legend,
                                                       lw=0.5, ax=ax)
                    # Labels
                    ax.set(xlabel='Wavelength (nm)', ylabel='Measurement',
                           title=f'Suspicious curve: ID {s}')
                    # Grid
                    ax.grid(which='both', lw=.25, color='black')
                    # Legend if required
                    if legend:
                        ax.legend(edgecolor='black', fancybox=False)
                    # Config
                    sns.despine()
                    # Save if required
                    if plot_path:
                        plt.savefig(plot_path / f'suspicious_id_{s}.jpg')
                    plt.show()

            # Calculate the median curves of each ID
            median = (data
                      .groupby(groupby)
                      .transform('median', numeric_only=True)
                      .filter(regex='\d'))
            median.columns = median.columns.map(int)

            # Calculate mean absolute deviation (MAD) from the median curve
            mad = pd.DataFrame(data
                               .filter(items=self.wls)
                               .sub(median)
                               .abs()
                               .sum(axis=1)
                               .rename('mad'))

            # Add info
            mad = mad.join(data.filter(regex='\D'))

            # Keys with highest MAD
            mad_max = mad.groupby(groupby).idxmax()

            # Intersection of keys with higher MAD, and IDs that are suspicious
            self.suspicious = mad_max.loc[ids_over].mad.tolist()

            # Remove suspicious data
            if remove_suspicious:

                # Drop suspect spectra
                to_keep = self.rdata.index.difference(self.suspicious)
                filtered = self.rdata.loc[to_keep]

                # Print keys removed
                print('--------------------------')
                print('Suspicious curves removed:')
                for c in self.suspicious: print(c)
                print('--------------------------')

                # Create new Spectra object
                new = Spectra(filtered)

                # Repeat if required
                if recursive:
                    return new.consistency(std_threshold, measurements,
                                           groupby, remove_suspicious,
                                           plot_suspicious, recursive,
                                           legend, plot_path)

                # Or just return the new object
                else:
                    return new

            # In case suspicious are not to be removed
            else:
                return self

        # If suspicious stations do not exist
        else:
            self.suspicious = None
            print('--------------------')
            print('No suspicious curves')
            print('--------------------')
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
            self.rdata.loc[~self.is_ref]
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

    # -------------------------------------------------------------------------
    # Method: particulate absorption
    # -------------------------------------------------------------------------
    def tr(self, unit='percent', wl_offset=800,
           trans_pattern='T', wl_range=None):

        #--------------
        # Prepare data
        #--------------

        # Get data
        data = self.rdata.set_index('id')

        if unit == 'percent':
            data[self.wls] /= 100

        # Set filters
        fref = self.is_ref.to_numpy()
        fsample = self.is_sample.to_numpy()
        ftotal = self.is_total.to_numpy()
        ftrans = (self.config == trans_pattern).to_numpy()

        # Apply the filters to separate data
        trans_total = data.loc[fsample & ftrans & ftotal].filter(regex='\d')
        trans_depig = data.loc[fsample & ftrans & ~ftotal].filter(regex='\d')
        refle_total = data.loc[fsample & ~ftrans & ftotal].filter(regex='\d')
        refle_depig = data.loc[fsample & ~ftrans & ~ftotal].filter(regex='\d')

        # Get references
        Tref = data.loc[fref & ftrans].filter(regex='\d').mean()
        Rref = data.loc[fref & ~ftrans].filter(regex='\d').mean()

        #---------------
        # Apply offsets
        #---------------

        # Transmittance cannot exceed the references
        trans_blank_offset = Tref[wl_offset] - trans_total[wl_offset]
        trans_blank_offset.loc[trans_blank_offset > 0] = 0

        # Update transmittance
        trans_total = trans_total.add(trans_blank_offset, axis=0)

        # Total and depigmented curves must match at the reference wavelength
        trans_offset = trans_total[wl_offset] - trans_depig[wl_offset]
        refle_offset = refle_total[wl_offset] - refle_depig[wl_offset]

        # Update tables
        trans_depig = trans_depig.add(trans_offset, axis=0)
        refle_depig = refle_depig.add(refle_offset, axis=0)

        #-----------------------
        # Plot of offsetted data
        #-----------------------

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(5.75, 5.75/3), dpi=300,
                                 sharey=False, constrained_layout=False)

        # Plot transmittance
        trans_total.transpose().plot(legend=False, lw=.2,
                                     color='blue', ax=axes[0])
        trans_depig.transpose().plot(legend=False, lw=.2,
                                     color='red', ax=axes[0])
        Tref.plot(color='black', lw=.5, ax=axes[0])

        # Configure
        axes[0].set(title='Transmittance')

        # Plot reflectance
        refle_total.transpose().plot(legend=False, lw=.2,
                                     color='blue', ax=axes[1])
        refle_depig.transpose().plot(legend=False, lw=.2,
                                     color='red', ax=axes[1])
        Rref.plot(color='black', lw=.5, ax=axes[1])

        # Configure
        axes[1].set(title='Reflectance')
        sns.despine()

        #-----------------
        # Blank correction
        #-----------------

        # Apply correction (sample/reference)
        Tt = trans_total.div(Tref, axis=1)
        Td = trans_depig.div(Tref, axis=1)
        Rt = refle_total.div(Rref, axis=1)
        Rd = refle_depig.div(Rref, axis=1)

        #----------------
        # Tau calculation
        #----------------

        # Optical depth of transmittance
        odt_total = np.log10(1 / Tt)
        odt_depig = np.log10(1 / Td)

        # Quantity used to calculate tau
        odts_total = odt_total.sub(.5 * odt_total[750], axis=0)
        odts_depig = odt_depig.sub(.5 * odt_depig[750], axis=0)

        # Correction factor tau
        tau_total = 1.15 - 0.17 * odts_total
        tau_total[(odts_total <= .02) | (odts_total >= .7)] = 1

        tau_depig = 1.15 - 0.17 * odts_depig
        tau_depig[(odts_depig <= .02) | (odts_depig >= .7)] = 1

        #------------------------------------
        # Correction and absorption retrieval
        #------------------------------------

        # Calculate absorptances
        absorptance_total = ((1 - Tt + Rref * (Tt - Rt)) /
                             (1 + Rref * Tt * tau_total))

        absorptance_depig = ((1 - Td + Rref * (Td - Rd)) /
                             (1 + Rref * Td * tau_depig))

        # Calculate optical densities
        od_total = np.log10(1 / (1 - absorptance_total))
        od_depig = np.log10(1 / (1 - absorptance_depig))

        # Calculate absorption
        abs_total = np.log(10) * .719 * (od_total ** 1.2287)
        abs_depig = np.log(10) * .719 * (od_depig ** 1.2287)

        return abs_total, abs_depig
