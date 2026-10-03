'''
Module: particulate.py
Purpose: Particulate absorption retrieval from transmittance-reflectance data.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python: 3.11+

Dependencies: numpy, pandas, matplotlib, seaborn
Public API: ParticulateProcessor.get_absorption
'''

import numpy as np
import pandas as pd

from .plots import (plot_part_absorption_absorbance,
                    plot_part_absorption_raw)


class ParticulateProcessor:
    '''
    Particulate absorption processor associated with a Spectra instance.
    '''

    def __init__(self, spectra):
        '''
        Create a particulate processor attached to a Spectra object.
        '''
        self.spectra = spectra

    def get_absorption(self,
                       vol_diameter,
                       percentual=False,
                       type_measurement='absorbance',
                       wl_offset=800,
                       use_tau=True,
                       trans_pattern='T',
                       wl_range=None,
                       plot_raw=False,
                       plot_absorbance=False,
                       split_total_depig_blank=True,
                       custom_blanks=None):
        '''
        Retrieve particulate absorption from TR measurements.

        Parameters
        ----------
        vol_diameter : tuple or pandas.DataFrame
            Either ``(volume, diameter)`` scalars or an indexed table with
            ``volume`` and ``diameter`` columns.
        percentual : bool, optional
            Whether input transmittance and reflectance are percentages.
        type_measurement : {'absorbance', 'tr'}, optional
            Whether the raw spectra contain absorbance or T/R values.
        wl_offset : int, optional
            Wavelength used to align and offset the curves.
        use_tau : bool, optional
            Whether to apply the tau correction.
        split_total_depig_blank : bool, optional
            Whether to calculate separate total and depigmented blank means.
            If ``False``, all blanks for each measurement mode are combined
            into one mean and used for both correction paths.
        wl_range : tuple or None, optional
            Inclusive wavelength interval to process.
        plot_raw, plot_absorbance : bool, optional
            Whether to display intermediate diagnostic plots.
        custom_blanks : pandas.DataFrame or None, optional
            Optional blank measurements to use instead of blanks in the data. It must
            be provided in the same ``type_measurement``.
            The index must contain exactly one ``T`` row and one ``R`` row,
            and the columns must be wavelength labels. When provided,
            ``split_total_depig_blank`` must be ``False``.

        Returns
        -------
        Spectra
            The associated object, with ``part_abs_total`` and
            ``part_abs_depig`` attributes.
        '''
        spectra = self.spectra

        # Convert input measurements to decimal transmittance/reflectance.
        if type_measurement == 'absorbance':
            data = np.power(10, -spectra.raw_spectra.copy())
        elif type_measurement == 'tr':
            data = spectra.raw_spectra.copy()
        else:
            raise ValueError('type_measurement must be either "absorbance" or "tr"')

        # Restrict the analysis to the requested wavelength range
        if wl_range is not None:
            wli, wlf = wl_range
            if wli not in spectra.wls or wlf not in spectra.wls:
                raise ValueError('wl_range values out of bounds.')
            data = data.loc[:, wli:wlf]

        # Convert to decimal if the input is in percentage
        if percentual:
            data /= 100

        # Update wavelengths
        wls = data.columns

        # Separate blank and sample measurements by mode and pigmentation
        fblank = spectra.is_blank.to_numpy()
        fsample = spectra.is_sample.to_numpy()
        ftotal = spectra.is_total.to_numpy()
        ftrans = (spectra.rmode == trans_pattern).to_numpy()

        # Validate and prepare externally supplied blank measurements.
        custom_blank_data = None
        if custom_blanks is not None:
            if split_total_depig_blank:
                raise ValueError(
                    'split_total_depig_blank must be False when '
                    'custom_blanks is provided.')
            if not isinstance(custom_blanks, pd.DataFrame):
                raise TypeError('custom_blanks must be a pandas DataFrame.')
            if custom_blanks.empty:
                raise ValueError('custom_blanks must not be empty.')

            blank_modes = pd.Index(custom_blanks.index).map(
                lambda mode: str(mode).upper())
            if not blank_modes.isin(['T', 'R']).all():
                raise ValueError(
                    'custom_blanks index must contain only T and R labels.')
            if (blank_modes == 'T').sum() != 1 or (blank_modes == 'R').sum() != 1:
                raise ValueError(
                    'custom_blanks must contain exactly one T row and one R row.')

            custom_blank_data = custom_blanks.copy()
            try:
                numeric_wavelengths = np.asarray(
                    [float(wavelength) for wavelength in custom_blank_data.columns],
                    dtype=float)
                if not np.equal(numeric_wavelengths,
                                numeric_wavelengths.astype(int)).all():
                    raise ValueError
                custom_blank_data.columns = pd.Index(
                    numeric_wavelengths.astype(int), dtype='int64')
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    'custom_blanks columns must be integer wavelengths.') from error

            missing_wavelengths = wls.difference(custom_blank_data.columns)
            if not missing_wavelengths.empty:
                raise ValueError(
                    'custom_blanks is missing wavelengths: '
                    f'{missing_wavelengths.tolist()}')
            custom_blank_data = custom_blank_data.loc[:, wls]
            custom_blank_data.index = blank_modes
            custom_blank_data = custom_blank_data.apply(pd.to_numeric,
                                                        errors='coerce')
            if custom_blank_data.isna().any().any():
                raise ValueError(
                    'custom_blanks must contain only numeric measurements.')
            if type_measurement == 'absorbance':
                custom_blank_data = np.power(10, -custom_blank_data)
            if percentual:
                custom_blank_data /= 100

        trans_total_raw = data.loc[fsample & ftrans & ftotal].sort_index()
        refl_total_raw = data.loc[fsample & ~ftrans & ftotal].sort_index()
        trans_depig_raw = data.loc[fsample & ftrans & ~ftotal].sort_index()
        refl_depig_raw = data.loc[fsample & ~ftrans & ~ftotal].sort_index()

        # Calculate the blank reference curves used for correction.
        if custom_blank_data is not None:
            Tblank_total = custom_blank_data.loc['T']
            Rblank_total = custom_blank_data.loc['R']
            Tblank_depig = Tblank_total.copy()
            Rblank_depig = Rblank_total.copy()
        elif split_total_depig_blank:
            Tblank_total = data.loc[fblank & ftrans & ftotal].mean()
            Rblank_total = data.loc[fblank & ~ftrans & ftotal].mean()
            Tblank_depig = data.loc[fblank & ftrans & ~ftotal].mean()
            Rblank_depig = data.loc[fblank & ~ftrans & ~ftotal].mean()
        else:
            Tblank_total = data.loc[fblank & ftrans].mean()
            Rblank_total = data.loc[fblank & ~ftrans].mean()
            Tblank_depig = Tblank_total.copy()
            Rblank_depig = Rblank_total.copy()

        # Reuse the available blank type when only one pigmentation class is
        # present in the reference measurements.
        if Tblank_total.empty and not Tblank_depig.empty:
            Tblank_total = Tblank_depig.copy()
            print('Warning: Transmittance blank total is missing; '
                  'using depigmented blank instead.')
        elif Tblank_depig.empty and not Tblank_total.empty:
            Tblank_depig = Tblank_total.copy()
            print('Warning: Transmittance blank depigmented is missing; '
                  'using total blank instead.')

        if Rblank_total.empty and not Rblank_depig.empty:
            Rblank_total = Rblank_depig.copy()
            print('Warning: Reflectance blank total is missing; '
                  'using depigmented blank instead.')
        elif Rblank_depig.empty and not Rblank_total.empty:
            Rblank_depig = Rblank_total.copy()
            print('Warning: Reflectance blank depigmented is missing; '
                  'using total blank instead.')

        # Align total and depigmented curves at the reference wavelength
        if wl_offset not in wls:
            raise ValueError('wl_offset out of bounds.')

        # Ensure that total transmittance is not higher than the blank at any
        # wavelength by applying each sample's greatest positive difference.
        trans_blank_offset = (trans_total_raw.sub(Tblank_total, axis=1)
                              .max(axis=1)
                              .clip(lower=0))
        trans_total = trans_total_raw.sub(trans_blank_offset, axis=0)

        # Ensure that total reflectance is not higher than the blank at any
        # wavelength by applying each sample's greatest positive difference.
        refl_blank_offset = (refl_total_raw.sub(Rblank_total, axis=1)
                             .max(axis=1)
                             .clip(lower=0))
        refl_total = refl_total_raw.sub(refl_blank_offset, axis=0)

        # Ensure that depigmented curves equal total ones at the reference wavelength
        trans_offset = trans_depig_raw[wl_offset] - trans_total[wl_offset].to_numpy()
        trans_depig = trans_depig_raw.sub(trans_offset, axis=0)

        refl_offset = refl_depig_raw[wl_offset] - refl_total[wl_offset].to_numpy()
        refl_depig = refl_depig_raw.sub(refl_offset, axis=0)

        # Plot raw and offset curves if requested
        if plot_raw:
            plot_part_absorption_raw(
                trans_total_raw, trans_depig_raw,
                refl_total_raw, refl_depig_raw,
                trans_total, trans_depig, refl_total, refl_depig,
                Tblank_total, Tblank_depig, Rblank_total, Rblank_depig,
                spectra.id)

        # Unique IDs are used to reindex the corrected curves and ensure
        # alignment of dataframes
        unique_ids = spectra.id.loc[trans_total.index].to_numpy()

        # Correct the transmittance and reflectance curves by the blank reference
        Tt = trans_total.div(Tblank_total, axis=1).set_index(unique_ids)
        Td = trans_depig.div(Tblank_depig, axis=1).set_index(unique_ids)
        Rt = refl_total.div(Rblank_total, axis=1).set_index(unique_ids)
        Rd = refl_depig.div(Rblank_depig, axis=1).set_index(unique_ids)

        # Apply the optional tau correction to transmittance optical depth
        if use_tau:
            if 750 not in data.columns:
                raise ValueError('Wavelength 750 is required when use_tau=True')
            odt_total = np.log10(1 / Tt)
            odt_depig = np.log10(1 / Td)
            odts_total = odt_total.sub(.5 * odt_total[750], axis=0)
            odts_depig = odt_depig.sub(.5 * odt_depig[750], axis=0)
            tau_total = 1.15 - 0.17 * odts_total
            tau_total[(odts_total <= .02) | (odts_total >= .7)] = 1
            tau_depig = 1.15 - 0.17 * odts_depig
            tau_depig[(odts_depig <= .02) | (odts_depig >= .7)] = 1
        else:
            tau_total = tau_depig = 1

        # Retrieve total and depigmented particulate absorbance.
        absorbance_total = ((1 - Tt + Rblank_total * (Tt - Rt)) /
                            (1 + Rblank_total * Tt * tau_total))
        absorbance_depig = ((1 - Td + Rblank_depig * (Td - Rd)) /
                            (1 + Rblank_depig * Td * tau_depig))
        absorbance_total = absorbance_total.fillna(0).clip(lower=0)
        absorbance_depig = absorbance_depig.fillna(0).clip(lower=0)

        spectra.absorbance_total = absorbance_total
        spectra.absorbance_depig = absorbance_depig

        if plot_absorbance:
            plot_part_absorption_absorbance(
                absorbance_total, absorbance_depig, spectra.id)

        # Convert absorbance to optical density before the volume and area correction
        od_total = np.log10(1 / (1 - absorbance_total))
        od_depig = np.log10(1 / (1 - absorbance_depig))

        # Build the filter-area/filtered-volume normalization factor.
        if isinstance(vol_diameter, tuple) and len(vol_diameter) == 2:
            volume, diameter = vol_diameter
            normalization = np.pi * diameter ** 2 / (4 * volume)
        elif isinstance(vol_diameter, pd.DataFrame):
            required = {'volume', 'diameter'}
            if not required.issubset(vol_diameter.columns):
                raise ValueError(
                    'vol_diameter must contain volume and diameter columns.')
            dimensions = vol_diameter.reindex(od_total.index)
            if dimensions[['volume', 'diameter']].isna().any().any():
                raise ValueError(
                    'vol_diameter must have volume and diameter values for '
                    'every sample id.')
            volume = pd.to_numeric(dimensions['volume'], errors='coerce')
            diameter = pd.to_numeric(dimensions['diameter'], errors='coerce')
            normalization = np.pi * diameter ** 2 / (4 * volume)
        else:
            raise TypeError(
                'vol_diameter must be a (volume, diameter) tuple or a '
                'DataFrame with volume and diameter columns.')

        if np.any(np.asarray(normalization) <= 0):
            raise ValueError('volume and diameter must be greater than zero.')

        spectra.part_abs_total = np.log(10) * .719 * (od_total ** 1.2287)
        spectra.part_abs_depig = np.log(10) * .719 * (od_depig ** 1.2287)
        spectra.part_abs_total = spectra.part_abs_total.mul(normalization, axis=0)
        spectra.part_abs_depig = spectra.part_abs_depig.mul(normalization, axis=0)
        return spectra
