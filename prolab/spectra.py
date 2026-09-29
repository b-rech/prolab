# -*- coding: utf-8 -*-
'''
Script: spectra.py
Description: Shared spectral data container and processor composition.
Author: Bruno Rech
Institution: INPE
Created: 2026-03-12
Python version: 3.11

Dependencies:
    - pandas
    - prolab.cdom
    - prolab.consistency
    - prolab.particulate

Public API:
    - Spectra
    - spectra.cdom.get_absorption()
    - spectra.particulate.get_absorption()
    - spectra.consistency.check()

Usage:
    python example.py
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
        # Keep the input table and identify numeric wavelength columns.
        self.raw_data = data
        self.raw_spectra = self.raw_data.filter(regex='\d')
        self.wls = self.raw_spectra.columns.map(int).to_numpy()
        self.raw_spectra.columns = self.wls

        # Expose metadata columns as attributes used by the processors.
        meta_cols = data.columns.difference(self.raw_spectra.columns)
        for col in meta_cols:
            setattr(self, col, data[col])

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
