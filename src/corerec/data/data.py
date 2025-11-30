# Copyright 2023 The CoreRec Authors. All Rights Reserved.
# ===========================================================
import copy
import os
import pickle
import warnings
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix


def get_rng(seed=None):
    """Get a random number generator."""
    return np.random.default_rng(seed)


def estimate_batches(data_size, batch_size):
    """Estimate number of batches for a given data size and batch size."""
    return int(np.ceil(data_size / batch_size))


def validate_format(fmt, supported_formats):
    """Validate if the format is supported."""
    if fmt not in supported_formats:
        raise ValueError(f"Format {fmt} is not supported. Supported formats: {supported_formats}")
    return fmt


class BaseDataset:
    """Base class for all datasets.

    Parameters
    ----------
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = get_rng(seed)
        self.ignored_attrs = []

    def reset(self):
        """Reset the random number generator for reproducibility."""
        self.rng = get_rng(self.seed)
        return self

    def save(self, fpath):
        """Save the dataset to a file."""
        # Ensure the directory exists if a directory is specified
        dirname = os.path.dirname(fpath)
        if dirname:  # Only create directories if a directory is provided
            os.makedirs(dirname, exist_ok=True)
        with open(fpath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fpath):
        """Load a dataset from a file."""
        with open(fpath, "rb") as f:
            dataset = pickle.load(f)
        return dataset


class ContextualDataset(BaseDataset):
    """Dataset with contextual information (e.g., time, location, device).

    Parameters
    ----------
    data: List[Tuple]
        List of tuples (user, item, rating, context).
    context_features: Dict
        Dictionary of contextual features.
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, data, context_features, seed=None):
        super().__init__(seed)
        self.data = data
        self.context_features = context_features
        self.uid_map = OrderedDict()
        self.iid_map = OrderedDict()
        self.map_ids()

    def map_ids(self):
        """Map user and item IDs to internal indices."""
        for uid, iid, _, _ in self.data:
            self.uid_map.setdefault(uid, len(self.uid_map))
            self.iid_map.setdefault(iid, len(self.iid_map))

    def get_context(self, user, item):
        """Get contextual features for a user-item pair."""
        return self.context_features.get((user, item), {})


class MultimodalDataset(BaseDataset):
    """Dataset with multimodal features (e.g., text, images).

    Parameters
    ----------
    data: List[Tuple]
        List of tuples (user, item, rating).
    text_data: Dict, optional, default: None
        Dictionary of text features.
    image_data: Dict, optional, default: None
        Dictionary of image features.
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, data, text_data=None, image_data=None, seed=None):
        super().__init__(seed)
        self.data = data
        self.text_data = text_data
        self.image_data = image_data
        self.uid_map = OrderedDict()
        self.iid_map = OrderedDict()
        self.map_ids()

    def map_ids(self):
        """Map user and item IDs to internal indices."""
        for uid, iid, _ in self.data:
            self.uid_map.setdefault(uid, len(self.uid_map))
            self.iid_map.setdefault(iid, len(self.iid_map))

    def get_text_features(self, item):
        """Get text features for an item."""
        return self.text_data.get(item, None)

    def get_image_features(self, item):
        """Get image features for an item."""
        return self.image_data.get(item, None)


class TemporalDataset(BaseDataset):
    """Dataset with temporal information (e.g., timestamps).

    Parameters
    ----------
    data: List[Tuple]
        List of tuples (user, item, rating, timestamp).
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, data, seed=None):
        super().__init__(seed)
        self.data = data
        self.uid_map = OrderedDict()
        self.iid_map = OrderedDict()
        self.timestamps = []
        self.map_ids()

    def map_ids(self):
        """Map user and item IDs to internal indices."""
        for uid, iid, _, timestamp in self.data:
            self.uid_map.setdefault(uid, len(self.uid_map))
            self.iid_map.setdefault(iid, len(self.iid_map))
            self.timestamps.append(timestamp)

    def get_temporal_data(self):
        """Get temporal data (user, item, timestamp)."""
        return [(self.uid_map[uid], self.iid_map[iid], t) for uid, iid, _, t in self.data]


class GraphDataset(BaseDataset):
    """Dataset with graph-based information (e.g., adjacency matrix).

    Parameters
    ----------
    data: List[Tuple]
        List of tuples (user, item, rating).
    adjacency_matrix: scipy.sparse.csr_matrix
        Adjacency matrix representing the graph.
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, data, adjacency_matrix, seed=None):
        super().__init__(seed)
        self.data = data
        self.adjacency_matrix = adjacency_matrix
        self.uid_map = OrderedDict()
        self.iid_map = OrderedDict()
        self.map_ids()

    def map_ids(self):
        """Map user and item IDs to internal indices."""
        for uid, iid, _ in self.data:
            self.uid_map.setdefault(uid, len(self.uid_map))
            self.iid_map.setdefault(iid, len(self.iid_map))

    def get_graph_data(self):
        """Get graph data (adjacency matrix)."""
        return self.adjacency_matrix


class SequentialDataset(BaseDataset):
    """Dataset with sequential information (e.g., session-based data).

    Parameters
    ----------
    data: List[Tuple]
        List of tuples (user, session, item, rating).
    seed: int, optional, default: None
        Random seed for reproducibility.
    """

    def __init__(self, data, seed=None):
        super().__init__(seed)
        self.data = data
        self.uid_map = OrderedDict()
        self.sid_map = OrderedDict()
        self.iid_map = OrderedDict()
        self.map_ids()

    def map_ids(self):
        """Map user, session, and item IDs to internal indices."""
        for uid, sid, iid, _ in self.data:
            self.uid_map.setdefault(uid, len(self.uid_map))
            self.sid_map.setdefault(sid, len(self.sid_map))
            self.iid_map.setdefault(iid, len(self.iid_map))

    def get_sequential_data(self):
        """Get sequential data (user, session, item)."""
        return [
            (self.uid_map[uid], self.sid_map[sid], self.iid_map[iid])
            for uid, sid, iid, _ in self.data
        ]


# Example Usage
# if __name__ == "__main__":
#     # Contextual Dataset
#     contextual_data = [("user1", "item1", 5, {"time": "morning", "location": "NY"})]
#     contextual_features = {("user1", "item1"): {"time": "morning", "location": "NY"}}
#     contextual_dataset = ContextualDataset(contextual_data, contextual_features)

#     # Multimodal Dataset
#     multimodal_data = [("user1", "item1", 5)]
#     text_features = {"item1": "This is a great product!"}
#     image_features = {"item1": "image1.jpg"}
#     multimodal_dataset = MultimodalDataset(multimodal_data, text_features, image_features)

#     # Temporal Dataset
#     temporal_data = [("user1", "item1", 5, 1672531200)]  # (user, item, rating, timestamp)
#     temporal_dataset = TemporalDataset(temporal_data)

#     # Graph Dataset
#     graph_data = [("user1", "item1", 5)]
#     adjacency_matrix = csr_matrix(([1], ([0], [1])), shape=(2, 2))  # Example adjacency matrix
#     graph_dataset = GraphDataset(graph_data, adjacency_matrix)

#     # Sequential Dataset
#     sequential_data = [("user1", "session1", "item1", 5)]
#     sequential_dataset = SequentialDataset(sequential_data)

#     # Save and load example
#     temporal_dataset.save("temporal_dataset.pkl")
#     loaded_dataset = BaseDataset.load("temporal_dataset.pkl")
#     print("Loaded dataset timestamps:", loaded_dataset.timestamps)
