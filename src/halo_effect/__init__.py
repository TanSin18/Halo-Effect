"""
Halo Effect Analysis - Core Package

Modules for analyzing retail store closure halo effects using:
- DBSCAN clustering for trade area identification
- Prophet time series forecasting for COVID-19 impact analysis
- Census data integration for demographic normalization
"""

__version__ = "1.0.0"
__author__ = "SK0759"

from . import baby_trade_area
from . import bbby_trade_area
from . import closed_stores_trade_area
from . import forecasting_covid_halo

__all__ = [
    'baby_trade_area',
    'bbby_trade_area',
    'closed_stores_trade_area',
    'forecasting_covid_halo',
]
