#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for training a multimodal recommendation model.

This script demonstrates how to use the multimodal components in CoreRec
to train a recommendation model that combines text and image data.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
from PIL import Image

from corerec.multimodal.fusion_model import MultimodalFusionModel
from corerec.data.multimodal_dataset import (
    MultimodalRecommendationDataset,
    collate_multimodal_batch,
)
from corerec.trainer.trainer import Trainer
from corerec.trainer.callbacks import EarlyStopping, ModelCheckpoint
from corerec.trainer.metrics import get_metrics_dict
from corerec.utils.config import load_config
from corerec.utils.logging import setup_logging
from corerec.utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a multimodal recommendation model")

    # Configuration
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing data files"
    )
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model parameters
    parser.add_argument(
        "--fusion_type", type=str, default=None, help="Fusion type (concat, attention, gating)"
    )
    parser.add_argument("--use_images", action="store_true", help="Use image features")
    parser.add_argument("--use_text", action="store_true", help="Use text features")

    return parser.parse_args()


def load_data(args, config):
    """Load data for training.

    Args:
        args: Command line arguments
        config (Dict): Configuration dictionary

    Returns:
        Tuple: Training data, validation data, item texts, item image paths
    """
    data_config = config.get("data", {})

    # Get file paths
    interactions_path = os.path.join(
        args.data_dir, data_config.get("interactions_file", "interactions.csv")
    )
    item_metadata_path = os.path.join(
        args.data_dir, data_config.get("item_metadata_file", "item_metadata.json")
    )
    image_dir = os.path.join(args.data_dir, data_config.get("image_dir", "images"))

    # Load interactions
    interactions = pd.read_csv(interactions_path)

    # Load item metadata
    with open(item_metadata_path, "r") as f:
        item_metadata = json.load(f)

    # Extract item texts and image paths
    item_texts = {}
    item_image_paths = {}

    for item_id, metadata in item_metadata.items():
        if "description" in metadata:
            item_texts[item_id] = metadata["description"]

        if "image" in metadata:
            image_path = os.path.join(image_dir, metadata["image"])
            if os.path.exists(image_path):
                item_image_paths[item_id] = image_path

    # Split data into train and validation
    train_ratio = data_config.get("train_ratio", 0.8)
    np.random.seed(args.seed)

    # Shuffle interactions
    interactions = interactions.sample(frac=1).reset_index(drop=True)

    # Split based on ratio
    split_idx = int(len(interactions) * train_ratio)
    train_interactions = interactions.iloc[:split_idx]
    val_interactions = interactions.iloc[split_idx:]

    return train_interactions, val_interactions, item_texts, item_image_paths


def create_datasets(
    train_interactions, val_interactions, item_texts, item_image_paths, args, config
):
    """Create datasets for training.

    Args:
        train_interactions (pd.DataFrame): Training interactions
        val_interactions (pd.DataFrame): Validation interactions
        item_texts (Dict): Item text descriptions
        item_image_paths (Dict): Item image paths
        args: Command line arguments
        config (Dict): Configuration dictionary

    Returns:
        Tuple: Training dataset, validation dataset
    """
    data_config = config.get("data", {})

    # Create image transform
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create training dataset
    train_dataset = MultimodalRecommendationDataset(
        interactions=train_interactions,
        item_texts=item_texts if args.use_text else {},
        item_image_paths=item_image_paths if args.use_images else {},
        image_transform=image_transform,
        num_negatives=data_config.get("num_negatives", 4),
        user_id_col=data_config.get("user_id_col", "user_id"),
        item_id_col=data_config.get("item_id_col", "item_id"),
        rating_col=data_config.get("rating_col", "rating"),
        use_text_features=args.use_text,
        use_image_features=args.use_images,
        preload_images=data_config.get("preload_images", True),
    )

    # Create validation dataset
    val_dataset = MultimodalRecommendationDataset(
        interactions=val_interactions,
        item_texts=item_texts if args.use_text else {},
        item_image_paths=item_image_paths if args.use_images else {},
        image_transform=image_transform,
        num_negatives=data_config.get("num_negatives", 4),
        user_id_col=data_config.get("user_id_col", "user_id"),
        item_id_col=data_config.get("item_id_col", "item_id"),
        rating_col=data_config.get("rating_col", "rating"),
        use_text_features=args.use_text,
        use_image_features=args.use_images,
        preload_images=data_config.get("preload_images", True),
    )

    return train_dataset, val_dataset


def create_model(args, config):
    """Create the multimodal model.

    Args:
        args: Command line arguments
        config (Dict): Configuration dictionary

    Returns:
        MultimodalFusionModel: The multimodal model
    """
    model_config = config.get("model", {})

    # Override fusion type if provided in args
    if args.fusion_type:
        model_config["fusion_type"] = args.fusion_type

    # Create model
    model = MultimodalFusionModel(
        name=model_config.get("name", "multimodal_model"), config=model_config
    )

    return model


def train_model(model, train_dataset, val_dataset, args, config):
    """Train the multimodal model.

    Args:
        model (MultimodalFusionModel): The multimodal model
        train_dataset (MultimodalRecommendationDataset): Training dataset
        val_dataset (MultimodalRecommendationDataset): Validation dataset
        args: Command line arguments
        config (Dict): Configuration dictionary
    """
    train_config = config.get("training", {})

    # Create data loaders
    batch_size = args.batch_size or train_config.get("batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
        collate_fn=collate_multimodal_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        collate_fn=collate_multimodal_batch,
    )

    # Create optimizer
    learning_rate = args.learning_rate or train_config.get("learning_rate", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create scheduler
    scheduler = None
    if train_config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    # Create callbacks
    callbacks = [
        EarlyStopping(patience=train_config.get("patience", 10), monitor="val_loss"),
        ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "multimodal_model_epoch_{epoch:02d}.pt"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Create metrics
    metrics = get_metrics_dict(k=10)

    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        callbacks=callbacks,
        checkpoint_dir=args.model_dir,
        log_dir=args.log_dir,
    )

    # Train model
    epochs = args.epochs or train_config.get("epochs", 20)
    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, metrics=metrics)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.log_dir, "train_multimodal.log")
    setup_logging(log_file)
    logger = logging.getLogger("train_multimodal")

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Load data
    logger.info("Loading data...")
    train_interactions, val_interactions, item_texts, item_image_paths = load_data(args, config)
    logger.info(
        f"Loaded {len(train_interactions)} training interactions and {len(val_interactions)} validation interactions"
    )
    logger.info(
        f"Loaded {len(item_texts)} item descriptions and {len(item_image_paths)} item images"
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        train_interactions, val_interactions, item_texts, item_image_paths, args, config
    )
    logger.info(
        f"Created datasets with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(args, config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    logger.info("Training model...")
    train_model(model, train_dataset, val_dataset, args, config)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
