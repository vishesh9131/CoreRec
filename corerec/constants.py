"""
Default Column Names and Constants for CoreRec

These provide sensible defaults while allowing user overrides.
Keeps things consistent acrosss the framework without forcing
rigid patterns on users.
"""


# default column names for interaction data
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"


# similarity metric identifiers
SIM_COOCCURRENCE = "cooccurrence"
SIM_COSINE = "cosine"
SIM_JACCARD = "jaccard"
SIM_LIFT = "lift"
SIM_INCLUSION_INDEX = "inclusion_index"
SIM_MUTUAL_INFORMATION = "mutual_information"
SIM_LEXICOGRAPHERS_MI = "lexicographers_mi"


# all supported similarity types in one place
SUPPORTED_SIMILARITY_TYPES = [
    SIM_COOCCURRENCE,
    SIM_COSINE,
    SIM_JACCARD,
    SIM_LIFT,
    SIM_INCLUSION_INDEX,
    SIM_MUTUAL_INFORMATION,
    SIM_LEXICOGRAPHERS_MI,
]
