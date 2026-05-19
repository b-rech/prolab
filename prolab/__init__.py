'''
ProLab
------
Processing of ocean color laboratory measurements.
'''

from .spectra import Spectra
from .io import read_files

__all__ = [
    'Spectra',
    'read_files',
]

__version__ = '0.1.0'