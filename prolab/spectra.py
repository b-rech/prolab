'''
Module: spectra.py
Purpose: Shared spectral data container and processor composition.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python: 3.11+

Dependencies: pandas, prolab.cdom, prolab.consistency, prolab.correction,
prolab.particulate

Public API: Spectra, spectra.cdom.get_absorption(),
spectra.particulate.get_absorption(), spectra.consistency.check()
'''

# %% Dependencies

import pandas as pd

from .cdom import CDOMProcessor
from .consistency import ConsistencyProcessor
from .correction import BreakMethods
from .particulate import ParticulateProcessor


# %% Class definition

class Spectra:
    """Class to handle laboratory spectral measurements."""

    def __init__(self, data: 'pd.DataFrame'):
        """Create a Spectra container and attach its processing modules.

        Parameters
        ----------
        data : pandas.DataFrame
            Table containing an ``id`` column, metadata, and wavelength
            columns whose names contain numeric values.
        """
        # Keep an independent table and normalize wavelength column names.
        self.raw_data = data.copy()
        wavelength_columns = self.raw_data.filter(regex=r'\d').columns
        wavelength_values = pd.Index(
            wavelength_columns.map(float).map(int),
            dtype='int64',
        )
        self.raw_data.rename(
            columns=dict(zip(wavelength_columns, wavelength_values)),
            inplace=True,
        )
        self.raw_spectra = self.raw_data[wavelength_values]
        self.raw_spectra.columns = wavelength_values
        self.wls = wavelength_values.to_numpy()

        # Expose metadata columns as attributes used by the processors.
        meta_cols = self.raw_data.columns.difference(self.raw_spectra.columns)
        for col in meta_cols:
            setattr(self, col, self.raw_data[col])

        # Domain-specific processors operate on this shared object.
        self.cdom = CDOMProcessor(self)
        self.consistency = ConsistencyProcessor(self)
        self.correction = BreakMethods(self)
        self.particulate = ParticulateProcessor(self)

    def __repr__(self):
        """Return a compact representation with observation dimensions."""
        return (f'Spectra(observations={len(self.raw_spectra)}, '
                f'wavelengths={len(self.wls)})')

    def __str__(self):
        """Return a human-readable description of the object."""
        return (f'Object Spectra with {len(self.raw_spectra)} observations and '
                f'{len(self.wls)} wavelengths')
