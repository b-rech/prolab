"""
Module: cdom.py
Purpose: CDOM absorption correction and exponential curve fitting.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python: 3.11+

Dependencies: numpy, pandas, scipy
Public API: CDOMProcessor.get_absorption, CDOMProcessor.expfit
"""

import numpy as np
import pandas as pd
import scipy.optimize


class CDOMProcessor:
    """CDOM processing associated with a Spectra instance."""

    def __init__(self, spectra):
        """Create a CDOM processor attached to a :class:`Spectra` object."""
        self.spectra = spectra

    def __getattr__(self, name):
        """Read processor state from the associated Spectra object."""
        return getattr(self.spectra, name)

    def __setattr__(self, name, value):
        """Store processing results on the associated Spectra object."""
        if name == 'spectra':
            object.__setattr__(self, name, value)
        else:
            setattr(self.spectra, name, value)

    def get_absorption(self,
                       blank=None,
                       wl_null_point_correction=600,
                       pathlength=.1,
                       mean_replicates=True):
        """
        Retrieve absorption from absorbance curves.

        Parameters
        ----------
        blank : bool, pd.DataFrame, or None, optional
            If `False`, no blank correction is applied. If `True`, the mean
            blank curves in the dataset are used. If `None`, the mean blank
            curves are used when available; otherwise a zero blank is assumed.
            A DataFrame uses its wavelength columns as the blank reference.
        wl_null_point_correction : int, tuple, or None, optional
            The wavelength or interval at which the absorption is offset.
        pathlength : float, optional
            The instrument optical path length in meters.
        mean_replicates : bool, optional
            Whether to average replicate measurements before blank correction.

        Returns
        -------
        prolab.Spectra
            The same object, with a new `absorption` attribute.
        """
        if pathlength <= 0:
            raise ValueError('pathlength must be greater than zero')

        # Select the blank source: no correction, existing blanks, or a
        # caller-provided reference table.
        if blank is None:
            blank = bool(self.is_blank.any())

        if blank is False:
            print('No blank correction will be applied')
            blank = pd.DataFrame({wl: 0 for wl in self.wls}, index=['blank'])

        elif blank is True:
            print('Blank correction will be applied with existing curves')
            if not self.is_blank.any():
                raise ValueError('No blank curves were found in the data.')
            blank = (self.raw_spectra.loc[self.is_blank.to_numpy()]
                     .mean()
                     .rename('blank')
                     .to_frame()
                     .transpose())

        elif isinstance(blank, pd.DataFrame):
            print('Blank correction will be applied with provided curves')
            if blank.empty:
                raise ValueError('The provided blank DataFrame is empty.')
            blank = blank.copy()
            blank.columns = blank.columns.map(int)

        else:
            raise TypeError(
                'blank must be one of: None, True, False, or a pandas.DataFrame.'
            )

        if blank.columns.empty:
            raise ValueError('The blank dataset does not contain wavelength columns.')

        # Restrict samples and blank curves to their common wavelength range.
        wlmin = max(self.wls.min(), blank.columns.min())
        wlmax = min(self.wls.max(), blank.columns.max())
        wl_range = range(wlmin, wlmax + 1)

        # Average replicate samples when requested; otherwise keep each row.
        if mean_replicates:
            samples = (self.raw_data.loc[self.is_sample]
                       .groupby('id')
                       .mean(numeric_only=True)
                       .filter(regex='\d'))
        else:
            samples = self.raw_spectra.loc[self.is_sample.to_numpy()]

        if samples.empty:
            raise ValueError('No sample curves were found in the data.')

        samples.columns = samples.columns.map(int)
        self.raw_samples = samples[wl_range]
        blank = blank[wl_range]

        # Apply either one common blank or the closest blank for each sample.
        if len(blank) == 1:
            self.absorbance = self.raw_samples.sub(blank.squeeze(), axis=1)
        else:
            blank_ref = blank.loc[:, blank.columns >= 685]
            if blank_ref.empty:
                raise ValueError(
                    'The blank DataFrame must include wavelengths >= 685 nm '
                    'to select the closest blank curve.'
                )

            corr_dict = {}
            blank_dict = {}
            blank_sal_dict = {}

            for station, curve in self.raw_samples.iterrows():
                ref_candidates = blank_ref.loc[:, blank_ref.columns >= 685]
                blank_sel = (abs(ref_candidates.sub(curve.loc[685:], axis=1))
                             .mean(axis=1)
                             .sort_values()
                             .index[0])
                corr_dict[station] = curve - blank.loc[blank_sel]
                blank_dict[station] = blank.loc[blank_sel]
                blank_sal_dict[station] = blank_sel

            self.absorbance = pd.DataFrame(corr_dict).transpose()
            self.blank_curves = pd.DataFrame(blank_dict).transpose()
            self.blank_salinity = pd.DataFrame(blank_sal_dict, index=[0])

        # Remove the null-point offset at one wavelength or over an interval.
        check_offset = True
        if isinstance(wl_null_point_correction, (int, float)):
            if wl_null_point_correction not in self.absorbance.columns:
                raise ValueError(
                    f'Wavelength {wl_null_point_correction} '
                    'is not available in the absorbance data.'
                )
            self.npc_offset = self.absorbance[wl_null_point_correction]
            absorbance_corr = self.absorbance.sub(self.npc_offset, axis=0)

        elif isinstance(wl_null_point_correction, (list, tuple, np.ndarray)):
            if len(wl_null_point_correction) != 2:
                raise ValueError(
                    'wl_null_point_correction must be a pair (wli, wlf) '
                    'or a single wavelength.'
                )
            wli, wlf = wl_null_point_correction
            if wli > wlf:
                raise ValueError(
                    'The null point interval must satisfy wli <= wlf.'
                )
            self.npc_offset = self.absorbance.loc[:, wli:wlf].mean(axis=1)
            absorbance_corr = self.absorbance.sub(self.npc_offset, axis=0)

        elif wl_null_point_correction is None:
            absorbance_corr = self.absorbance
            check_offset = False

        else:
            raise ValueError('Set a proper parameter to wl_null_point_correction')

        if check_offset and (self.npc_offset > .0015).any():
            print('Warning: there are offsets > 0.0015 AU in the null '
                  'point correction; check the npc_offset attribute.')

        # Convert corrected absorbance to absorption coefficients.
        self.absorption = 2.303 * absorbance_corr / pathlength
        return self

    def expfit(self, wl_ref=440, wl_range=(350, 500)):
        """Fit an exponential curve and store its parameters on the spectra.

        Parameters
        ----------
        wl_ref : int, optional
            Reference wavelength for the fitted absorption coefficient.
        wl_range : tuple, list, array-like, or None, optional
            Inclusive fitting interval. ``None`` uses all wavelengths.

        Returns
        -------
        Spectra
            The associated object, with a ``cdom_fitted`` attribute.
        """
        if not hasattr(self, 'absorption'):
            raise AttributeError(
                'The absorption data are missing. Run `get_absorption` '
                'before calling `expfit`.'
            )

        if wl_range is None:
            wli, wlf = self.wls.min(), self.wls.max()
        elif isinstance(wl_range, (list, tuple, np.ndarray)):
            wli, wlf = map(int, wl_range)
            if wli > wlf:
                raise ValueError(
                    'The wavelength interval must satisfy wli <= wlf.'
                )
            if wli not in self.wls or wlf not in self.wls:
                raise ValueError(
                    'The requested wavelength range is outside bounds.'
                )
        else:
            raise TypeError(
                'wl_range must be a tuple, list, array-like, or None.'
            )

        if wl_ref not in self.wls:
            raise ValueError(
                f'Wavelength {wl_ref} is not available in the absorbance data.'
            )

        # Fit A(lambda) = A(ref) * exp(-slope * (lambda - ref)).
        def exp_curve(wl, abs_ref, slope):
            return abs_ref * np.exp(-slope * (wl - wl_ref))

        ydata = self.absorption if wl_range is None else self.absorption.loc[:, wli:wlf]
        x = ydata.columns.map(int).to_numpy(dtype=float)
        abs_key = f'a{wl_ref:.0f}'
        slope_key = f'slope_{wli:.0f}_{wlf:.0f}'
        fit_dict = {'id': [], abs_key: [], slope_key: []}

        # Fit each station independently and collect its parameters.
        for station, y in ydata.iterrows():
            y = pd.to_numeric(y, errors='coerce').to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                raise ValueError(
                    f'No valid absorbance values were found for sample {station}.'
                )
            try:
                coeffs, _ = scipy.optimize.curve_fit(
                    exp_curve, x, y, (0.1, 0.02), maxfev=20000)
            except RuntimeError as exc:
                raise ValueError(
                    f'Curve fitting failed for sample {station} in the interval '
                    f'[{wli}, {wlf}] nm.'
                ) from exc
            if not np.all(np.isfinite(coeffs)):
                raise ValueError(
                    f'Non-finite fit parameters were returned for sample {station}.'
                )
            fit_dict['id'].append(station)
            fit_dict[abs_key].append(coeffs[0])
            fit_dict[slope_key].append(coeffs[1])

        self.cdom_fitted = pd.DataFrame(fit_dict).set_index('id')
        return self
