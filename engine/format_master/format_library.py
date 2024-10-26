import warnings

def format_warning(message):
    """Format the warning message to be more visually appealing."""
    return f"\033[93m[WARNING]\033[0m {message}"  # Yellow text

# Attempt to import optional libraries and handle ImportError
try:
    import xml.etree.ElementTree as ET
except ImportError:
    warnings.warn(format_warning("xml.etree.ElementTree is not available. XML files will not be supported."))

try:
    import yaml
except ImportError:
    warnings.warn(format_warning("PyYAML is not available. YAML files will not be supported."))

try:
    import pyarrow.parquet as pq
except ImportError:
    warnings.warn(format_warning("PyArrow is not available. Parquet files will not be supported."))

try:
    import h5py
except ImportError:
    warnings.warn(format_warning("h5py is not available. HDF5 files will not be supported."))

try:
    import sqlite3
except ImportError:
    warnings.warn(format_warning("sqlite3 is not available. SQLite databases will not be supported."))

try:
    import feather
except ImportError:
    warnings.warn(format_warning("Feather format is not available. Feather files will not be supported."))

try:
    import avro.datafile
    import avro.io
    import avro.schema
except ImportError:
    warnings.warn(format_warning("Avro is not available. Avro files will not be supported."))

# Additional imports for new formats
try:
    import pandas.io.parsers
except ImportError:
    warnings.warn(format_warning("pandas.io.parsers is not available. Fixed-width files will not be supported."))

try:
    import pystan
except ImportError:
    warnings.warn(format_warning("Pystan is not available. Protocol Buffers files will not be supported."))

try:
    import msgpack
except ImportError:
    warnings.warn(format_warning("msgpack is not available. MessagePack files will not be supported."))

try:
    import pyorc
except ImportError:
    warnings.warn(format_warning("pyorc is not available. ORC files will not be supported."))

try:
    import toml
except ImportError:
    warnings.warn(format_warning("toml is not available. TOML files will not be supported."))

try:
    import sqlalchemy
except ImportError:
    warnings.warn(format_warning("SQLAlchemy is not available. SQL Server and PostgreSQL exports will not be supported."))

try:
    import pymongo
except ImportError:
    warnings.warn(format_warning("PyMongo is not available. MongoDB exports will not be supported."))

try:
    import requests
    from google.oauth2 import service_account
    import gspread
except ImportError:
    warnings.warn(format_warning("gspread or google-auth is not available. Google Sheets will not be supported."))

try:
    import textract
except ImportError:
    warnings.warn(format_warning("Textract is not available. LaTeX files will not be supported."))

def check_feather_support():
    try:
        import pyarrow.feather
    except ImportError:
        warnings.warn(format_warning("Feather format is not available. Feather files will not be supported."))

def check_avro_support():
    try:
        import fastavro
    except ImportError:
        warnings.warn(format_warning("Avro is not available. Avro files will not be supported."))

def check_pystan_support():
    try:
        import pystan
    except ImportError:
        warnings.warn(format_warning("Pystan is not available. Protocol Buffers files will not be supported."))

def check_pyorc_support():
    try:
        import pyorc
    except ImportError:
        warnings.warn(format_warning("pyorc is not available. ORC files will not be supported."))

def check_pymongo_support():
    try:
        import pymongo
    except ImportError:
        warnings.warn(format_warning("PyMongo is not available. MongoDB exports will not be supported."))

def check_gspread_support():
    try:
        import gspread
        import google.auth
    except ImportError:
        warnings.warn(format_warning("gspread or google-auth is not available. Google Sheets will not be supported."))

def check_textract_support():
    try:
        import textract
    except ImportError:
        warnings.warn(format_warning("Textract is not available. LaTeX files will not be supported."))

# Example usage in a function
def process_feather_file(file_path):
    check_feather_support()
    # Your code to process the feather file

def process_avro_file(file_path):
    check_avro_support()
    # Your code to process the avro file

# Add similar functions for other formats
