"""
This module provides functions to load, preprocess, and validate data from various file formats for use in a recommendation system.
## Author: Vishesh Yadav

Functions:
- register_format_detector(detector): Decorator to register a format detector.
- load_data(file_path): Loads data from a file, supporting multiple formats such as CSV, JSON, Excel, and more.
- load_xml(file_path): Loads data from an XML file into a DataFrame.
- load_hdf5(file_path): Loads data from an HDF5 file into a DataFrame.
- load_hdf4(file_path): Loads data from an HDF4 file into a DataFrame.
- load_avro(file_path): Loads data from an Avro file into a DataFrame.
- load_sqlite(file_path): Loads data from a SQLite database into a DataFrame.
- load_orc(file_path): Loads data from an ORC file into a DataFrame.
- load_sql_export(file_path): Placeholder for loading data from SQL Server or PostgreSQL exports.
- load_bson(file_path): Loads data from a BSON file (MongoDB export) into a DataFrame.
- load_latex(file_path): Loads data from a LaTeX table into a DataFrame.
- load_access_db(file_path): Loads data from an Access database into a DataFrame.
- detect_format(df): Detects the format of the dataset, such as interaction list or matrix.
- preprocess_data(df, data_format): Preprocesses data based on its detected format.
- validate_data(df): Validates the data for use in a recommendation system.
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
import logging

from corerec.format_master.format_library import *

# Registry for format detection functions
FORMAT_DETECTORS = []

def register_format_detector(detector):
    """Decorator to register a format detector."""
    FORMAT_DETECTORS.append(detector)
    return detector

def load_data(file_path):
    """Load data from various formats."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.csv':
        # Specify dtypes to avoid dtype inference issues
        dtype_spec = {
            'isbn': 'object',
            'isbn13': 'object',
            'original_publication_year': 'float64',
            'average_rating': 'float64',
            'ratings_count': 'int64'
        }
        return pd.read_csv(file_path, dtype=dtype_spec)
    
    elif file_extension == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    
    elif file_extension == '.json':
        with open(file_path, 'r') as f:
            return pd.DataFrame(json.load(f))
    
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path, sheet_name=None)  # Load all sheets as a dict of DataFrames
    
    elif file_extension == '.parquet' and 'pyarrow.parquet' in globals():
        return pd.read_parquet(file_path)
    
    elif file_extension == '.xml' and 'ET' in globals():
        return load_xml(file_path)
    
    elif file_extension == '.dat':
        return pd.read_csv(file_path, delimiter='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    
    elif file_extension == '.yaml' and 'yaml' in globals():
        with open(file_path, 'r') as f:
            return pd.DataFrame(yaml.safe_load(f))
    
    elif file_extension == '.h5' and 'h5py' in globals():
        return load_hdf5(file_path)
    
    elif file_extension == '.hdf4' and 'h5py' in globals():
        return load_hdf4(file_path)
    
    elif file_extension == '.feather' and 'feather' in globals():
        return pd.read_feather(file_path)
    
    elif file_extension == '.avro' and 'avro' in globals():
        return load_avro(file_path)
    
    elif file_extension in ['.db', '.sqlite']:
        return load_sqlite(file_path)
    
    elif file_extension == '.tsv' and 'pandas.io.parsers' in globals():
        return pd.read_fwf(file_path)
    
    elif file_extension == '.dta':
        if 'pandas.io.parsers' in globals():
            return pd.read_stata(file_path)
        else:
            warnings.warn("pandas.io.parsers is not available. Stata files will not be supported.")
    
    elif file_extension == '.sav':
        if 'pandas.io.parsers' in globals():
            return pd.read_spss(file_path)
        else:
            warnings.warn("pandas.io.parsers is not available. SPSS files will not be supported.")
    
    elif file_extension == '.pkl':
        return pd.read_pickle(file_path)
    
    elif file_extension == '.msgpack' and 'msgpack' in globals():
        return pd.read_msgpack(file_path)
    
    elif file_extension == '.pb' and 'pystan' in globals():
        # Example Protocol Buffers loader (custom implementation needed)
        warnings.warn("Protocol Buffers loading is not implemented.")
        return pd.DataFrame()
    
    elif file_extension == '.orc' and 'pyorc' in globals():
        return load_orc(file_path)
    
    elif file_extension == '.fwf':
        return pd.read_fwf(file_path)
    
    elif file_extension == '.geojson':
        return pd.read_json(file_path)
    
    elif file_extension == '.toml' and 'toml' in globals():
        with open(file_path, 'r') as f:
            data = toml.load(f)
            return pd.DataFrame(data)
    
    elif file_extension == '.html':
        return pd.read_html(file_path)
    
    elif file_extension == '.sql' and 'sqlalchemy' in globals():
        return load_sql_export(file_path)
    
    elif file_extension == '.bson' and 'pymongo' in globals():
        return load_bson(file_path)
    
    elif file_extension in ['.tex'] and 'textract' in globals():
        return load_latex(file_path)
    
    elif file_extension in ['.md']:
        warnings.warn("Markdown files are not supported.")
        return pd.DataFrame()
    
    elif file_extension in ['.sas7bdat']:
        if 'pandas.io.parsers' in globals():
            return pd.read_sas(file_path, format='sas7bdat')
        else:
            warnings.warn("pandas.io.parsers is not available. SAS files will not be supported.")
    
    elif file_extension in ['.accdb', '.mdb']:
        if 'pyodbc' in globals():
            return load_access_db(file_path)
        else:
            warnings.warn("pyodbc is not available. Access DB files will not be supported.")
    
    elif file_extension in ['.tex']:
        if 'textract' in globals():
            return load_latex(file_path)
        else:
            warnings.warn("Textract is not available. LaTeX files will not be supported.")
    
    else:
        warnings.warn(f"Unsupported file format: {file_extension}")
        return pd.DataFrame()

def load_xml(file_path):
    """Load data from an XML file."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for child in root:
        row = {}
        for subchild in child:
            row[subchild.tag] = subchild.text
        data.append(row)
    return pd.DataFrame(data)

def load_hdf5(file_path):
    """Load data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = {key: np.array(f[key]) for key in f.keys()}
    return pd.DataFrame(data)

def load_hdf4(file_path):
    """Load data from an HDF4 file."""
    try:
        import pyhdf.SD as SD
        hdf = SD.SD(file_path)
        data = {}
        for sds in hdf:
            data[sds.name()] = sds.get()
        return pd.DataFrame(data)
    except ImportError:
        warnings.warn("pyhdf is not available. HDF4 files will not be supported.")
        return pd.DataFrame()

def load_avro(file_path):
    """Load data from an Avro file."""
    try:
        with open(file_path, 'rb') as f:
            reader = avro.datafile.DataFileReader(f, avro.io.DatumReader())
            data = [record for record in reader]
            reader.close()
        return pd.DataFrame(data)
    except Exception as e:
        warnings.warn(f"Failed to load Avro file: {e}")
        return pd.DataFrame()

def load_sqlite(file_path):
    """Load data from a SQLite database."""
    try:
        conn = sqlite3.connect(file_path)
        query = "SELECT * FROM main_table"  # Replace with your table name
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        warnings.warn(f"Failed to load SQLite database: {e}")
        return pd.DataFrame()

def load_orc(file_path):
    """Load data from an ORC file."""
    try:
        import pyorc
        with open(file_path, 'rb') as f:
            reader = pyorc.Reader(f)
            columns = reader.schema.fields
            data = [dict(zip(columns, row)) for row in reader]
        return pd.DataFrame(data)
    except Exception as e:
        warnings.warn(f"Failed to load ORC file: {e}")
        return pd.DataFrame()

def load_sql_export(file_path):
    """Load data from SQL Server or PostgreSQL export."""
    try:
        # This is a placeholder. Implement actual SQL import as needed.
        warnings.warn("SQL export loading is not implemented.")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Failed to load SQL export: {e}")
        return pd.DataFrame()

def load_bson(file_path):
    """Load data from a BSON file (MongoDB export)."""
    try:
        import bson
        with open(file_path, 'rb') as f:
            data = bson.decode_all(f.read())
        return pd.DataFrame(data)
    except Exception as e:
        warnings.warn(f"Failed to load BSON file: {e}")
        return pd.DataFrame()

def load_latex(file_path):
    """Load data from a LaTeX table."""
    try:
        text = textract.process(file_path).decode('utf-8')
        # This is a simplistic parser; you may need a more robust solution
        return pd.read_csv(pd.compat.StringIO(text), sep='&', engine='python').replace('\\', '', regex=True)
    except Exception as e:
        warnings.warn(f"Failed to load LaTeX file: {e}")
        return pd.DataFrame()

def load_access_db(file_path):
    """Load data from an Access database."""
    try:
        import pyodbc
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            f'DBQ={file_path};'
        )
        conn = pyodbc.connect(conn_str)
        query = "SELECT * FROM main_table"  # Replace with your table name
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        warnings.warn(f"Failed to load Access DB: {e}")
        return pd.DataFrame()

@register_format_detector
def detect_interaction_list(df):
    """Detect if the format is an interaction list."""
    if {'user', 'item', 'rating'}.issubset(df.columns):
        return 'interaction_list'
    return None

@register_format_detector
def detect_implicit_feedback(df):
    """Detect if the format is implicit feedback."""
    if {'user', 'item'}.issubset(df.columns) and 'rating' not in df.columns:
        return 'implicit_feedback'
    return None

@register_format_detector
def detect_interaction_matrix(df):
    """Detect if the format is an interaction matrix."""
    if df.index.name == 'user' and all(isinstance(col, str) for col in df.columns):
        return 'interaction_matrix'
    return None

@register_format_detector
def detect_numerical_matrix(df):
    """Detect if the format is a numerical matrix."""
    if df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
        return 'numerical_matrix'
    return None

@register_format_detector
def detect_categorical_matrix(df):
    """Detect if the format is a categorical matrix."""
    if df.dtypes.apply(lambda x: np.issubdtype(x, np.object_)).all():
        return 'categorical_matrix'
    return None

@register_format_detector
def detect_time_series(df):
    """Detect if the format is a time series."""
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        return 'time_series'
    return None

@register_format_detector
def detect_geospatial(df):
    """Detect if the format contains geospatial data."""
    if {'latitude', 'longitude'}.issubset(df.columns):
        return 'geospatial'
    return None

@register_format_detector
def detect_user_user_matrix(df):
    """Detect if the format is a user-user matrix."""
    if df.index.name == 'user' and all(isinstance(col, str) for col in df.columns):
        return 'user_user_matrix'
    return None

@register_format_detector
def detect_user_item_matrix(df):
    """Detect if the format is a user-item matrix."""
    if 'user' in df.index.name and 'item' in df.columns:
        return 'user_item_matrix'
    return None

@register_format_detector
def detect_item_item_matrix(df):
    """Detect if the format is an item-item matrix."""
    if df.index.name == 'item' and all(isinstance(col, str) for col in df.columns):
        return 'item_item_matrix'
    return None

@register_format_detector
def detect_user_item_list(df):
    """Detect if the format is a user-item list."""
    if {'user', 'item'}.issubset(df.columns):
        return 'user_item_list'
    return None

@register_format_detector
def detect_user_item_rating_list(df):
    """Detect if the format is a user-item-rating list."""
    if {'user', 'item', 'rating'}.issubset(df.columns):
        return 'user_item_rating_list'
    return None

@register_format_detector
def detect_time_series(df):
    """Detect if the format is a time series."""
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        return 'time_series'
    return None

@register_format_detector
def detect_geospatial(df):
    """Detect if the format contains geospatial data."""
    if {'latitude', 'longitude'}.issubset(df.columns):
        return 'geospatial'
    return None

@register_format_detector
def detect_text_data(df):
    """Detect if the format contains text data."""
    if 'text' in df.columns or 'document' in df.columns:
        return 'text_data'
    return None

@register_format_detector
def detect_image_data(df):
    """Detect if the format contains image data."""
    if 'image_path' in df.columns or 'image_url' in df.columns:
        return 'image_data'
    return None

@register_format_detector
def detect_audio_data(df):
    """Detect if the format contains audio data."""
    if 'audio_path' in df.columns or 'audio_url' in df.columns:
        return 'audio_data'
    return None

@register_format_detector
def detect_video_data(df):
    """Detect if the format contains video data."""
    if 'video_path' in df.columns or 'video_url' in df.columns:
        return 'video_data'
    return None

@register_format_detector
def detect_implicit_feedback(df):
    """Detect if the format is implicit feedback."""
    if {'user', 'item'}.issubset(df.columns) and 'rating' not in df.columns:
        return 'implicit_feedback'
    return None

@register_format_detector
def detect_explicit_feedback(df):
    """Detect if the format is explicit feedback."""
    if {'user', 'item', 'rating'}.issubset(df.columns):
        return 'explicit_feedback'
    return None

@register_format_detector
def detect_contextual_data(df):
    """Detect if the format contains contextual data."""
    if {'context', 'user', 'item'}.issubset(df.columns):
        return 'contextual_data'
    return None

@register_format_detector
def detect_session_data(df):
    """Detect if the format contains session data."""
    if 'session_id' in df.columns:
        return 'session_data'
    return None

@register_format_detector
def detect_demographic_data(df):
    """Detect if the format contains demographic data."""
    if {'age', 'gender', 'location'}.issubset(df.columns):
        return 'demographic_data'
    return None

@register_format_detector
def detect_social_network_data(df):
    """Detect if the format contains social network data."""
    if {'user', 'friend'}.issubset(df.columns):
        return 'social_network_data'
    return None

@register_format_detector
def detect_purchase_history(df):
    """Detect if the format contains purchase history."""
    if {'user', 'item', 'purchase_date'}.issubset(df.columns):
        return 'purchase_history'
    return None

@register_format_detector
def detect_browsing_history(df):
    """Detect if the format contains browsing history."""
    if {'user', 'page', 'timestamp'}.issubset(df.columns):
        return 'browsing_history'
    return None

@register_format_detector
def detect_clickstream_data(df):
    """Detect if the format contains clickstream data."""
    if {'user', 'click', 'timestamp'}.issubset(df.columns):
        return 'clickstream_data'
    return None

@register_format_detector
def detect_search_queries(df):
    """Detect if the format contains search queries."""
    if {'user', 'query', 'timestamp'}.issubset(df.columns):
        return 'search_queries'
    return None

@register_format_detector
def detect_book_metadata(df):
    """Detect if the format is book metadata."""
    required_columns = {'title', 'authors', 'average_rating', 'isbn'}
    if required_columns.issubset(df.columns):
        return 'book_metadata'
    return None

# Add more detectors here up to 100 as needed

def detect_format(df):
    """Detect the format of the dataset."""
    for detector in FORMAT_DETECTORS:
        format_type = detector(df)
        if format_type:
            logging.info(f"Detected format: {format_type}")
            return format_type
    logging.warning("No matching format detected. Returning 'unknown'.")
    return 'unknown'

def preprocess_data(df, data_format):
    """Preprocess data based on its format."""
    if data_format == 'unknown':
        logging.error("Unknown data format. Preprocessing cannot proceed.")
        return None

    # Existing preprocessing logic...
    if data_format == 'interaction_list':
        if 'rating' in df.columns:
            df['rating'] = df['rating'] / df['rating'].max()
    elif data_format == 'implicit_feedback':
        df['interaction'] = 1
    elif data_format == 'interaction_matrix':
        df.fillna(0, inplace=True)
    elif data_format == 'numerical_matrix':
        df = (df - df.mean()) / df.std()
    elif data_format == 'categorical_matrix':
        df = pd.get_dummies(df)
    elif data_format == 'time_series':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif data_format == 'geospatial':
        df['geo'] = df.apply(lambda row: (row['latitude'], row['longitude']), axis=1)
    elif data_format == 'book_metadata':
        df['average_rating'] = df['average_rating'].fillna(df['average_rating'].mean())
        df['title'] = df['title'].fillna('Unknown Title')
    elif data_format == 'user_user_matrix':
        df['user'] = df['user'].astype(str)
        df.set_index('user', inplace=True)
    else:
        logging.warning(f"No preprocessing steps defined for format: {data_format}")
    # Add more preprocessing steps as needed
    return df

def validate_data(df):
    """Validate the data for the recommendation system."""
    if df.empty:
        warnings.warn("The DataFrame is empty.")
    if df.isnull().values.any():
        warnings.warn("The DataFrame contains missing values.")
    # Add more validation checks as needed
    pass

