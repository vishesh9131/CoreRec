from abc import ABC, abstractmethod
from typing import List, Optional
import os
import pickle
import json
import copy
import inspect
import warnings
import numpy as np
from datetime import datetime
from glob import glob

from corerec.sshh import *


class BaseCorerec(ABC):
    """Generic class for a recommender model. All recommendation models should inherit from this class.

    Parameters
    ----------------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    Attributes
    ----------
    num_users: int
        Number of users in training data.

    num_items: int
        Number of items in training data.

    total_users: int
        Number of users in training, validation, and test data.
        In other words, this includes unknown/unseen users.

    total_items: int
        Number of items in training, validation, and test data.
        In other words, this includes unknown/unseen items.

    uid_map: dict
        Global mapping of user ID-index.

    iid_map: dict
        Global mapping of item ID-index.

    max_rating: float
        Maximum value among the rating observations.

    min_rating: float
        Minimum value among the rating observations.

    global_mean: float
        Average value over the rating observations.
    """

    def __init__(self, name: str, trainable: bool = True, verbose: bool = False):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.is_fitted = False

        # attributes to be ignored when saving model
        self.ignored_attrs = ["train_set", "val_set", "test_set"]

        # useful information getting from train_set for prediction
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        self.max_rating = None
        self.min_rating = None
        self.global_mean = None

        self.__user_ids = None
        self.__item_ids = None

    @property
    def total_users(self):
        """Total number of users including users in test and validation if exists"""
        return len(self.uid_map) if self.uid_map is not None else self.num_users

    @property
    def total_items(self):
        """Total number of items including users in test and validation if exists"""
        return len(self.iid_map) if self.iid_map is not None else self.num_items

    @property
    def user_ids(self):
        """Return the list of raw user IDs"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return the list of raw item IDs"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

    def reset_info(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        ignored_attrs = set(self.ignored_attrs)
        for k, v in self.__dict__.items():
            if k in ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    @classmethod
    def _get_init_params(cls):
        """Get initial parameters from the model constructor"""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]
        return sorted([p.name for p in parameters])

    def clone(self, new_params=None):
        """Clone an instance of the model object."""
        new_params = {} if new_params is None else new_params
        init_params = {}
        for name in self._get_init_params():
            init_params[name] = new_params.get(name, copy.deepcopy(getattr(self, name)))
        return self.__class__(**init_params)

    def save(self, save_dir=None, save_trainset=False, metadata=None):
        """Save a recommender model to the filesystem."""
        if save_dir is None:
            return

        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        model_file = os.path.join(model_dir, "{}.pkl".format(timestamp))

        saved_model = copy.deepcopy(self)
        pickle.dump(saved_model, open(model_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            print("{} model is saved to {}".format(self.name, model_file))

        metadata = {} if metadata is None else metadata
        metadata["model_classname"] = type(saved_model).__name__
        metadata["model_file"] = os.path.basename(model_file)

        if save_trainset:
            trainset_file = model_file + ".trainset"
            pickle.dump(self.train_set, open(trainset_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            metadata["trainset_file"] = os.path.basename(trainset_file)

        with open(model_file + ".meta", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem."""
        if os.path.isdir(model_path):
            model_file = sorted(glob("{}/*.pkl".format(model_path)))[-1]
        else:
            model_file = model_path

        model = pickle.load(open(model_file, "rb"))
        model.trainable = trainable
        model.load_from = model_file
        return model

    @abstractmethod
    def fit(self, interaction_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the recommender system using the provided interaction matrix.
        
        Parameters:
        - interaction_matrix (scipy.sparse matrix): User-item interaction matrix.
        - user_ids (List[int]): List of user IDs.
        - item_ids (List[int]): List of item IDs.
        """
        pass

    @abstractmethod
    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Generate top-N item recommendations for a given user.
        
        Parameters:
        - user_id (int): The ID of the user.
        - top_n (int): The number of recommendations to generate.
        
        Returns:
        - List[int]: List of recommended item IDs.
        """
        pass

    def knows_user(self, user_idx):
        """Return whether the model knows user by its index"""
        return user_idx is not None and user_idx >= 0 and user_idx < self.num_users

    def knows_item(self, item_idx):
        """Return whether the model knows item by its index"""
        return item_idx is not None and item_idx >= 0 and item_idx < self.num_items

    def is_unknown_user(self, user_idx):
        """Return whether the model knows user by its index."""
        return not self.knows_user(user_idx)

    def is_unknown_item(self, item_idx):
        """Return whether the model knows item by its index."""
        return not self.knows_item(item_idx)

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set."""
        raise NotImplementedError()

    def early_stop(self, train_set, val_set, min_delta=0.0, patience=0):
        """Check if training should be stopped when validation loss has stopped improving."""
        self.current_epoch += 1
        current_value = self.monitor_value(train_set, val_set)
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            print("Early stopping:")
            print(f"- best epoch = {self.best_epoch}, stopped epoch = {self.stopped_epoch}")
            print(f"- best monitored value = {self.best_value:.6f} (delta = {current_value - self.best_value:.6f})")
            return True
        return False 