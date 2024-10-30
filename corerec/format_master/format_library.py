import logging

def format_info(message):
    """Format the info message to be more visually appealing."""
    return f"\033[94m[INFO]\033[0m {message}"  # Blue text

# Configure logging
logging.basicConfig(level=logging.INFO)

# Attempt to import optional libraries and handle ImportError
try:
    import xml.etree.ElementTree as ET
except ImportError:
    logging.info(format_info("xml.etree.ElementTree is not available. XML files will not be supported."))

try:
    import yaml
except ImportError:
    logging.info(format_info("PyYAML is not available. YAML files will not be supported."))

try:
    import pyarrow.parquet as pq
except ImportError:
    logging.info(format_info("PyArrow is not available. Parquet files will not be supported."))

try:
    import h5py
except ImportError:
    logging.info(format_info("h5py is not available. HDF5 files will not be supported."))

try:
    import sqlite3
except ImportError:
    logging.info(format_info("sqlite3 is not available. SQLite databases will not be supported."))

try:
    import feather
except ImportError:
    logging.info(format_info("Feather format is not available. Feather files will not be supported."))

try:
    import avro.datafile
    import avro.io
    import avro.schema
except ImportError:
    logging.info(format_info("Avro is not available. Avro files will not be supported."))

# Additional imports for new formats
try:
    import pandas.io.parsers
except ImportError:
    logging.info(format_info("pandas.io.parsers is not available. Fixed-width files will not be supported."))

try:
    import pystan
except ImportError:
    logging.info(format_info("Pystan is not available. Protocol Buffers files will not be supported."))

try:
    import msgpack
except ImportError:
    logging.info(format_info("msgpack is not available. MessagePack files will not be supported."))

try:
    import pyorc
except ImportError:
    logging.info(format_info("pyorc is not available. ORC files will not be supported."))

try:
    import toml
except ImportError:
    logging.info(format_info("toml is not available. TOML files will not be supported."))

try:
    import sqlalchemy
except ImportError:
    logging.info(format_info("SQLAlchemy is not available. SQL Server and PostgreSQL exports will not be supported."))

try:
    import pymongo
except ImportError:
    logging.info(format_info("PyMongo is not available. MongoDB exports will not be supported."))

try:
    import requests
    from google.oauth2 import service_account
    import gspread
except ImportError:
    logging.info(format_info("gspread or google-auth is not available. Google Sheets will not be supported."))

try:
    import textract
except ImportError:
    logging.info(format_info("Textract is not available. LaTeX files will not be supported."))

def check_feather_support():
    try:
        import pyarrow.feather
    except ImportError:
        logging.info(format_info("Feather format is not available. Feather files will not be supported."))

def check_avro_support():
    try:
        import fastavro
    except ImportError:
        logging.info(format_info("Avro is not available. Avro files will not be supported."))

def check_pystan_support():
    try:
        import pystan
    except ImportError:
        logging.info(format_info("Pystan is not available. Protocol Buffers files will not be supported."))

def check_pyorc_support():
    try:
        import pyorc
    except ImportError:
        logging.info(format_info("pyorc is not available. ORC files will not be supported."))

def check_pymongo_support():
    try:
        import pymongo
    except ImportError:
        logging.info(format_info("PyMongo is not available. MongoDB exports will not be supported."))

def check_gspread_support():
    try:
        import gspread
        import google.auth
    except ImportError:
        logging.info(format_info("gspread or google-auth is not available. Google Sheets will not be supported."))

def check_textract_support():
    try:
        import textract
    except ImportError:
        logging.info(format_info("Textract is not available. LaTeX files will not be supported."))

# Example usage in a function
def process_feather_file(file_path):
    check_feather_support()
    # Your code to process the feather file

def process_avro_file(file_path):
    check_avro_support()
    # Your code to process the avro file

# Add similar functions for other formats
