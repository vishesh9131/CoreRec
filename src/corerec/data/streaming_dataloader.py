"""
Streaming dataloader for efficient loading of large datasets.

This module provides a streaming dataloader that can handle large datasets
that don't fit in memory by loading data in chunks.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional, Callable, Iterator
import logging
import random
import os
import io
from collections import defaultdict


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large-scale recommendation data.

    This dataset reads data from files in chunks, making it suitable
    for datasets that don't fit in memory.

    Attributes:
        file_paths (List[str]): List of paths to data files
        batch_size (int): Size of batches to yield
        shuffle (bool): Whether to shuffle data
        transform (Optional[Callable]): Optional transform to apply to data
    """

    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 1024,
        shuffle: bool = True,
        chunk_size: int = 10000,
        transform: Optional[Callable] = None,
        user_id_col: str = "user_id",
        item_id_col: str = "item_id",
        rating_col: Optional[str] = "rating",
        file_format: str = "csv",
    ):
        """Initialize the streaming dataset.

        Args:
            file_paths (List[str]): List of paths to data files
            batch_size (int): Size of batches to yield
            shuffle (bool): Whether to shuffle data
            chunk_size (int): Number of rows to read at once
            transform (Optional[Callable]): Optional transform to apply to data
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            rating_col (Optional[str]): Column name for ratings
            file_format (str): Format of data files ('csv', 'parquet', 'json')
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.transform = transform
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col
        self.file_format = file_format.lower()

        # Check if file paths exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

    def _read_chunk(self, file_path: str) -> pd.DataFrame:
        """Read a chunk of data from a file.

        Args:
            file_path (str): Path to data file

        Returns:
            pd.DataFrame: Chunk of data
        """
        if self.file_format == "csv":
            return pd.read_csv(file_path, chunksize=self.chunk_size)
        elif self.file_format == "parquet":
            # For parquet, we read the whole file but process in chunks
            df = pd.read_parquet(file_path)
            return [df[i : i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        elif self.file_format == "json":
            return pd.read_json(file_path, lines=True, chunksize=self.chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def _process_chunk(self, chunk: pd.DataFrame) -> Iterator[Dict[str, torch.Tensor]]:
        """Process a chunk of data.

        Args:
            chunk (pd.DataFrame): Chunk of data

        Yields:
            Dict[str, torch.Tensor]: Sample from the dataset
        """
        # Create list of samples
        samples = []
        for _, row in chunk.iterrows():
            user_id = row[self.user_id_col]
            item_id = row[self.item_id_col]
            rating = (
                row[self.rating_col]
                if self.rating_col is not None and self.rating_col in row
                else 1.0
            )
            samples.append((user_id, item_id, rating))

        # Shuffle if needed
        if self.shuffle:
            random.shuffle(samples)

        # Yield samples
        batch = []
        for user_id, item_id, rating in samples:
            sample = {
                "user_id": user_id,
                "item_id": item_id,
                "rating": torch.tensor(rating, dtype=torch.float32),
            }

            # Apply transform if available
            if self.transform is not None:
                sample = self.transform(sample)

            batch.append(sample)

            if len(batch) >= self.batch_size:
                # Collate batch
                collated_batch = self._collate_batch(batch)
                yield collated_batch
                batch = []

        # Yield remaining samples
        if batch:
            collated_batch = self._collate_batch(batch)
            yield collated_batch

    def _collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch (List[Dict[str, Any]]): List of samples

        Returns:
            Dict[str, torch.Tensor]: Collated batch
        """
        # Initialize result
        result = defaultdict(list)

        # Collect values for each key
        for sample in batch:
            for key, value in sample.items():
                result[key].append(value)

        # Convert lists to tensors
        for key in result:
            if key in ["user_id", "item_id"] and not isinstance(result[key][0], torch.Tensor):
                # Convert IDs to tensor if they are not already
                try:
                    result[key] = torch.tensor(result[key], dtype=torch.long)
                except TypeError:
                    # Keep as list if conversion fails (e.g., string IDs)
                    pass
            elif isinstance(result[key][0], torch.Tensor):
                # Stack tensors
                result[key] = torch.stack(result[key])

        return dict(result)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Get an iterator over the dataset.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: Iterator over the dataset
        """
        # Shuffle file paths if needed
        if self.shuffle:
            random.shuffle(self.file_paths)

        # Iterate over files
        for file_path in self.file_paths:
            # Read chunks
            chunks = self._read_chunk(file_path)

            # Process chunks
            for chunk in chunks:
                yield from self._process_chunk(chunk)


class StreamingDataLoader:
    """
    Streaming data loader for large-scale recommendation data.

    This data loader wraps a StreamingDataset and provides an iterator
    over batches of data.

    Attributes:
        dataset (StreamingDataset): Dataset to load data from
        num_workers (int): Number of workers for loading data
    """

    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 1024,
        shuffle: bool = True,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
        user_id_col: str = "user_id",
        item_id_col: str = "item_id",
        rating_col: Optional[str] = "rating",
        file_format: str = "csv",
    ):
        """Initialize the streaming data loader.

        Args:
            file_paths (List[str]): List of paths to data files
            batch_size (int): Size of batches to yield
            shuffle (bool): Whether to shuffle data
            num_workers (int): Number of workers for loading data
            transform (Optional[Callable]): Optional transform to apply to data
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            rating_col (Optional[str]): Column name for ratings
            file_format (str): Format of data files ('csv', 'parquet', 'json')
        """
        self.dataset = StreamingDataset(
            file_paths=file_paths,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
            user_id_col=user_id_col,
            item_id_col=item_id_col,
            rating_col=rating_col,
            file_format=file_format,
        )
        self.num_workers = num_workers

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Get an iterator over batches of data.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: Iterator over batches of data
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process data loading
            return iter(self.dataset)
        else:
            # Multiple process data loading
            # Split file paths among workers
            per_worker = int(np.ceil(len(self.dataset.file_paths) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset.file_paths))

            # Create a dataset for this worker
            worker_dataset = StreamingDataset(
                file_paths=self.dataset.file_paths[start:end],
                batch_size=self.dataset.batch_size,
                shuffle=self.dataset.shuffle,
                transform=self.dataset.transform,
                user_id_col=self.dataset.user_id_col,
                item_id_col=self.dataset.item_id_col,
                rating_col=self.dataset.rating_col,
                file_format=self.dataset.file_format,
            )

            return iter(worker_dataset)
