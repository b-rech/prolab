# -*- coding: utf-8 -*-
'''
Script: spectra.py
Description: Spectra class and associated methods.
Author: Bruno Rech
Created: 2026-03-12
Python version: 3.11
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
# %%

# %%


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

        # Sampling stations
        self.stations = data.id[data.is_sample].unique().tolist()

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
            data = self.rdata.loc[self.is_blank].copy()

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
                                     .loc[data.id == s]
                                     .set_index('rep')
                                     .sort_index()
                                     .filter(regex='\d'))

                # Plot them if required
                if plot_suspicious:
                    fig, ax = plt.subplots(figsize=(5.75, 5.75/2), dpi=200)
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
    def get_absorption(self,
                       use_blank=True,
                       blank=None,
                       wl_null_point_correction=600,
                       pathlength=.1):
        '''
        Retrieves absorption from absorbance curves.

        Parameters
        ----------

        use_blank : bool, optional
            Indicates whether the data must be corrected using a blank.
            The default is `True`.

        blank : pd.DataFrame, optional
            Only used when `use_blank=True`.
            If `None`, the mean blank measurements
            within the data will be used.
            If a single curve is provided (wavelengths in the columns),
            it will be used in all measurements.
            If multiple curves are provided (salinity curves), the algorithm
            will select the blank closer to the sample at 685 nm, 
            following the IOCCG protocol.
            The default is `None`.

        wl_null_point_correction : int or array-like, optional
            Wavelength to be used at the null point correction (offset).
            If an integer is passed, the absorption at this wavelength will
            be used.
            If a tuple with the limits of a spectral interval is
            provided, the mean absorption within the range will be used.
            If `None`, the null point correction won't be executed.
            The default is `None`.

        pathlength : float, optional
            The instrument optical path length in meters. The default is `.1`.

        Returns
        -------
        prolab.Spectra
            The same object is returned.
            A new attribute `absorption` is added inplace.
        '''

        #------------------------
        # Prepare data and blank
        #------------------------

        # If the blank curves from the dataset are to be used
        if use_blank and blank is None:

            # Get mean reference curves
            blank = (self
                     .absorbance
                     .loc[self.is_blank]
                     .mean()
                     .rename('blank')
                     .to_frame()
                     .transpose())

        # If a blank curve (or curves) is provided
        elif use_blank and blank is not None:

            # Prepare the curve(s)
            blank.columns = blank.columns.map(int)

        # In case no blank is to be used (i.e., curves already corrected)
        else:
            blank = pd.DataFrame({wl: 0 for wl in self.wls}, index=['blank'])

        # Min and max wavelengths available
        wlmin = max(self.wls.min(), blank.columns.min())
        wlmax = min(self.wls.max(), blank.columns.max())
        wl_range = range(wlmin, wlmax + 1)

        # Get mean sample curves
        # They're grouped by id in case of multiple measurements
        mean = (self
                .rdata
                .loc[self.is_sample]
                .groupby('id')
                .mean(numeric_only=True)
                .filter(regex='\d'))
        mean.columns = mean.columns.map(int)

        # Filter wavelengths
        self.raw_absorbance = mean[wl_range]
        blank = blank[wl_range]

        #-----------------------
        # Correction with blank
        #-----------------------

        # For single blank curves
        if len(blank) == 1:
            self.absorbance = self.raw_absorbance.sub(blank.squeeze(), axis=1)

        # For multiple blank curves
        # It was designed to account for salinity curves, and the selection
        # is performed by minimizing the difference at 685 nm
        else:

            print('The correction will use the salinity curves provided')

            # Dictionary to receive corrected curves
            corr_dict = {}

            # Dictionaries to save reference curves and their names
            blank_dict = {}
            blank_sal_dict = {}

            # Iterate over curves
            for st, curve in self.raw_absorbance.iterrows():

                # Select a reference curve
                blank_sel = (
                    abs(blank.loc[:, 685:].sub(curve.loc[685:], axis=1))
                    .mean(axis=1)
                    .sort_values()
                    .index[0]
                    )

                # Subtract selected reference
                corr_dict[st] = (curve - blank.loc[blank_sel])

                # Save the selected reference curve
                blank_dict[st] = blank.loc[blank_sel]
                blank_sal_dict[st] = blank_sel

            # Corrected data
            self.absorbance = pd.DataFrame(corr_dict).transpose()

            # Reference used
            self.blank_curves = pd.DataFrame(blank_dict).transpose()
            self.blank_salinity = pd.DataFrame(blank_sal_dict, index=[0])

        #-----------------------
        # Null point correction
        #-----------------------

        _check = True

        # Single wavelength
        if isinstance(wl_null_point_correction, (int, float)):

            # Null point correction offsets
            self.npc_offset = self.absorbance[wl_null_point_correction]
            absorbance_off = self.absorbance.sub(self.npc_offset , axis=0)

        # Mean within an interval
        elif isinstance(wl_null_point_correction, (list, tuple, np.ndarray)):

            # Initial and final wavelengths
            wli, wlf = wl_null_point_correction

            # Offsets
            self.npc_offset = self.absorbance.loc[:, wli:wlf].mean(axis=1)
            absorbance_off = self.absorbance.sub(self.npc_offset , axis=0)

        # No correction
        elif wl_null_point_correction is None:
            absorbance_off = self.absorbance
            _check = False

        else:
            raise ValueError('Set a proper value to '
                             +'"wl_null_point_correction"')

        # Check according to IOCCG recommendation
        if _check and any(self.npc_offset > .0015):
            print('WARNING: there are offsets > 0.0015 AU in the null ' +
                  'point correction')

        # Calculate absorption
        self.absorption = 2.303 * absorbance_off / pathlength

        return self


    # -------------------------------------------------------------------------
    # Method: fit exponential curve
    # -------------------------------------------------------------------------
    def expfit(self, wl_ref=443, wl_range=(350, 500)):
        '''
        Fits exponential-decaying curves.

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
        '''

        # Exponential curve to fit
        def exp_curve(wl, abs_ref, slope):
            return abs_ref * np.exp(-slope * (wl - wl_ref))

        # Initial guesses for abs_ref and slope
        init_guess = (0.1, 0.02)

        # Curves to be fitted (y)
        if isinstance(wl_range, (list, tuple, np.ndarray)):
            ydata = self.absorption.loc[:, range(wl_range[0], wl_range[1])]
        else:
            ydata = self.absorption

        # Wavelengths (x)
        x = ydata.columns.map(int).to_numpy()

        # Dictionary to store data
        fit_dict = {'id': [], f'a{wl_ref:.0f}': [], 'slope': []}

        # Fit coefficients
        for st, y in ydata.iterrows():

            # Fit curve (least squares)
            coeffs, _ = scipy.optimize.curve_fit(exp_curve, x, y, init_guess)

            # Store data
            fit_dict['id'].append(st)
            fit_dict[f'a{wl_ref:.0f}'].append(coeffs[0])
            fit_dict['slope'].append(coeffs[1])

        # Final dataframe
        self.fitted = pd.DataFrame(fit_dict).set_index('id')

        return self

    # -------------------------------------------------------------------------
    # Method: particulate absorption
    # -------------------------------------------------------------------------
    def part_absorption(self,
                        vol_diameter,
                        unit='absorbance',
                        wl_offset=800,
                        use_tau=True,
                        trans_pattern='T',
                        wl_range=None,
                        plot_raw=False,
                        plot_absorbance=False):
        '''
        Retrieves particulate absorption based on the
        transmittance-reflectance method (Tassan & Ferrari, 1995).

        Parameters
        ----------
        unit : str, optional
            Indicates whether the measurements of transmittance and reflectance
            are in `decimal` (e.g., 0.20) or `percentual` form.
            The default is 'percentual'.
        wl_offset : int, optional
            At this wavelength, the transmittance of the sample is ensured to
            be lower than or equal to the transmittance of the reference.
            Also, at this wavelength the total and depigmented curves of a
            given sample are supposed to be equal, and depigmented spectra may
            be offset to ensure that. The default is 800.
        use_tau : bool, optional
            Indicates if the factor tau should be used in the calculation of
            the absorbance. The default is `True`.
        trans_pattern : str, optional
            The pattern (within the attribute `rmode`) used to identify
            transmittance spectra. The default is `T`.
        wl_range : array-like, optional
            If provided, the calculation are limited to this interval. Make
            sure that it includes the wavelength set in `wl_offset`.
            The default is None.
        plot_raw : bool, optional
            Whether to plot the raw curves after offset correction.
            The default is `False`.
        plot_absorbance : bool, optional
            Whether to plot the absorbance curves. The default is `False`.

        Returns
        -------
        abs_total : TYPE
            DESCRIPTION.
        abs_depig : TYPE
            DESCRIPTION.
        '''

        #--------------
        # Prepare data
        #--------------

        # Wavelengths
        wls = self.wls

        # Get data
        if unit == 'absorbance':
            rdata = self.rdata.set_index('id').filter(regex='\d')
            # Convert to transmittance/reflectance
            data = np.power(10, -rdata)
        elif unit == 'percentual':
            data[wls] /= 100

        # Get filtered volume and filter clearance area
        # vol = vol_diameter.volume
        # area = np.pi * (vol_diameter.diameter **2) / 4

        # Curtail wavelength range
        if wl_range is not None:
            data = data.loc[:, wl_range[0]:wl_range[1]]
            wls = np.arange(wl_range[0], wl_range[1] + 1)

        # Set filters
        fref = self.is_blank.to_numpy()
        fsample = self.is_sample.to_numpy()
        ftotal = self.is_total.to_numpy()
        ftrans = (self.rmode == trans_pattern).to_numpy()

        # Apply the filters to separate data
        trans_total = data.loc[fsample & ftrans & ftotal]
        trans_depig = data.loc[fsample & ftrans & ~ftotal]
        refle_total = data.loc[fsample & ~ftrans & ftotal]
        refle_depig = data.loc[fsample & ~ftrans & ~ftotal]

        # Get references
        Tblk_total = data.loc[fref & ftrans & ftotal].median()
        Rblk_total = data.loc[fref & ~ftrans & ftotal].median()
        Tblk_depig = data.loc[fref & ftrans & ~ftotal].median()
        Rblk_depig = data.loc[fref & ~ftrans & ~ftotal].median()

        #---------------
        # Apply offsets
        #---------------

        # Transmittance cannot exceed the blanks
        trans_blank_offset = Tblk_total[wl_offset] - trans_total[wl_offset]
        trans_blank_offset.loc[trans_blank_offset > 0] = 0

        # Update transmittance
        trans_total = trans_total.add(trans_blank_offset, axis=0)

        # Total and depigmented curves must match at the reference wavelength
        trans_offset = trans_total[wl_offset] - trans_depig[wl_offset]
        refle_offset = refle_total[wl_offset] - refle_depig[wl_offset]

        # Update tables
        trans_depig = trans_depig.add(trans_offset, axis=0)
        refle_depig = refle_depig.add(refle_offset, axis=0)

        #-----------------
        # Plot offset data
        #-----------------

        if plot_raw:

            # Long data
            plot_data = self.rdata.melt(
                id_vars=self.rdata.filter(regex='\D').columns.tolist()
                )
            plot_data['variable'] = plot_data.variable.map(int)
            plot_data['is_total'] = plot_data.is_total.replace(
                {True: 'Total', False: 'Extracted'})
            plot_data['is_trans'] = plot_data.is_trans.replace(
                {True: 'Transmittance', False: 'Reflectance'})

            # Styling
            hue_pal = {'Transmittance': 'r', 'Reflectance': 'b'}
            style_pal = {'Total': (None, None), 'Extracted': (3, 1)}

            # Iterate over stations
            for st in self.stations:

                # Create figure
                fig, ax = plt.subplots(figsize=(5.75, 5.75), dpi=100)

                # Get data
                _dt = plot_data.loc[(plot_data.id == st)]

                # Plot sample curves
                sns.lineplot(data=_dt, x='variable', y='value',
                             palette=hue_pal,
                             style='is_total',
                             style_order=['Total', 'Extracted'],
                             dashes=style_pal,
                             hue='is_trans',
                             hue_order=['Transmittance', 'Reflectance'],
                             ax=ax)

                # Plot blanks
                sns.lineplot(Tblk_total.transpose(), color='r', alpha=.5)
                sns.lineplot(Tblk_depig.transpose(), color='r', alpha=.5,
                             ls='dashed')
                sns.lineplot(Rblk_total.transpose(), color='b', alpha=.5)
                sns.lineplot(Rblk_depig.transpose(), color='b', alpha=.5,
                             ls='dashed')

                # Pega handles e labels
                handles, labels = ax.get_legend_handles_labels()
                
                # Cria legenda sem título
                ax.legend([handles[i] for i in [1, 2, 4, 5]],
                          [labels[i] for i in [1, 2, 4, 5]],
                          fontsize=8, ncols=2)

                # Config
                ax.set(title=st, xlabel='Wavelength (nm)',
                       ylabel='Transmittance | Reflectance')
                sns.despine()
                plt.show()


        #-----------------
        # Blank correction
        #-----------------

        # Apply correction (sample/reference)
        Tt = trans_total.div(Tblk_total, axis=1).copy()
        Td = trans_depig.div(Tblk_depig, axis=1).copy()
        Rt = refle_total.div(Rblk_total, axis=1).copy()
        Rd = refle_depig.div(Rblk_depig, axis=1).copy()

        #----------------
        # Tau calculation
        #----------------

        if use_tau:
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

        else:
            tau_total = 1
            tau_depig = 1

        #------------------------------------
        # Correction and absorption retrieval
        #------------------------------------

        # Calculate absorbances
        absorbance_total = ((1 - Tt + Rblk_total * (Tt - Rt)) /
                             (1 + Rblk_total * Tt * tau_total))

        absorbance_depig = ((1 - Td + Rblk_depig * (Td - Rd)) /
                             (1 + Rblk_depig * Td * tau_depig))

        if plot_absorbance:
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(5.75, 5.75/3), dpi=300,
                                     sharey=False, constrained_layout=False)

            # Plot total
            absorbance_total.transpose().plot(legend=False, lw=.2,
                                               color='blue', ax=axes[0])
            # Configure
            axes[0].set(title='Total')

            # Plot depigmented
            absorbance_depig.transpose().plot(legend=False, lw=.2,
                                               color='blue', ax=axes[1])
            # Configure
            axes[1].set(title='Depigmented')
            sns.despine()
            plt.show()

        # Calculate optical densities
        od_total = np.log10(1 / (1 - absorbance_total))
        od_depig = np.log10(1 / (1 - absorbance_depig))

        # # Calculate absorption
        abs_total = np.log(10) * .719 * (od_total ** 1.2287) #/ (vol / area)
        abs_depig = np.log(10) * .719 * (od_depig ** 1.2287) #/ (vol / area)

        return abs_total, abs_depig
