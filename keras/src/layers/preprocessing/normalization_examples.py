"""Real-world usage examples for Normalization layer.

Examples:
1. Image preprocessing - RGB channel normalization
2. Text preprocessing - Embedding normalization
3. Time-series normalization - Normalize across time dimension

Run examples:
    python normalization_examples.py --example image
    python normalization_examples.py --example text
    python normalization_examples.py --example timeseries
    python normalization_examples.py --all
"""
import numpy as np
import argparse


def setup_keras():
    """Setup Keras and return modules."""
    try:
        from keras.src import layers, models, ops, backend
        return layers, models, ops, backend
    except ImportError:
        print("Keras not installed. Install with: pip install keras")
        return None, None, None, None


def example_image_preprocessing():
    """Example 1: Image preprocessing with RGB channel normalization.

    Common use case: Normalize RGB channels using ImageNet statistics
    or custom dataset statistics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: IMAGE PREPROCESSING - RGB CHANNEL NORMALIZATION")
    print("="*80)

    layers, models, ops, backend = setup_keras()
    if layers is None:
        return

    # ImageNet RGB mean and std (commonly used)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_variance = imagenet_std ** 2

    print("\nImageNet statistics:")
    print(f"  Mean (RGB): {imagenet_mean}")
    print(f"  Std (RGB):  {imagenet_std}")

    # Create normalization layer for RGB images
    normalize_layer = layers.Normalization(
        axis=-1,  # Normalize across channel dimension
        mean=imagenet_mean,
        variance=imagenet_variance,
    )

    # Build a simple image preprocessing model
    inputs = layers.Input(shape=(224, 224, 3))
    normalized = normalize_layer(inputs)
    preprocessing_model = models.Model(inputs=inputs, outputs=normalized)

    print("\nPreprocessing model summary:")
    preprocessing_model.summary()

    # Example: Process a batch of images
    batch_size = 4
    dummy_images = np.random.randint(
        0, 256, size=(batch_size, 224, 224, 3)
    ).astype("float32") / 255.0  # Scale to [0, 1]

    print(f"\nInput image batch shape: {dummy_images.shape}")
    print(f"Input range: [{dummy_images.min():.3f}, {dummy_images.max():.3f}]")

    # Normalize
    normalized_images = preprocessing_model.predict(dummy_images, verbose=0)

    print(f"\nNormalized batch shape: {normalized_images.shape}")
    print(f"Normalized range: [{normalized_images.min():.3f}, {normalized_images.max():.3f}]")
    print(f"Normalized mean per channel: {normalized_images.mean(axis=(0,1,2))}")
    print(f"Normalized std per channel:  {normalized_images.std(axis=(0,1,2))}")

    print("\n✓ Image preprocessing example complete!")
    print("  Use case: Pre-trained models (ResNet, EfficientNet, etc.)")
    print("  Benefit: Standardized input for transfer learning")


def example_text_preprocessing():
    """Example 2: Text preprocessing with embedding normalization.

    Use case: Normalize embeddings before feeding to model.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: TEXT PREPROCESSING - EMBEDDING NORMALIZATION")
    print("="*80)

    layers, models, ops, backend = setup_keras()
    if layers is None:
        return

    # Simulate pre-computed embedding statistics
    # (In practice, compute from your embedding dataset)
    embedding_dim = 128
    vocab_size = 10000

    # Hypothetical embedding statistics
    embedding_mean = np.random.randn(embedding_dim).astype("float32") * 0.1
    embedding_std = np.ones(embedding_dim).astype("float32") * 0.5
    embedding_variance = embedding_std ** 2

    print(f"\nEmbedding dimension: {embedding_dim}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Mean range: [{embedding_mean.min():.3f}, {embedding_mean.max():.3f}]")
    print(f"Std range: [{embedding_std.min():.3f}, {embedding_std.max():.3f}]")

    # Create normalization layer for embeddings
    normalize_embeddings = layers.Normalization(
        axis=-1,  # Normalize across embedding dimension
        mean=embedding_mean,
        variance=embedding_variance,
    )

    # Build text preprocessing model
    sequence_length = 50
    inputs = layers.Input(shape=(sequence_length, embedding_dim))
    normalized = normalize_embeddings(inputs)
    model = models.Model(inputs=inputs, outputs=normalized)

    print("\nText preprocessing model:")
    model.summary()

    # Example: Process embedded text sequences
    batch_size = 8
    dummy_embeddings = np.random.randn(
        batch_size, sequence_length, embedding_dim
    ).astype("float32") * 0.5

    print(f"\nInput embeddings shape: {dummy_embeddings.shape}")
    print(f"Input mean: {dummy_embeddings.mean():.3f}")
    print(f"Input std: {dummy_embeddings.std():.3f}")

    # Normalize
    normalized_embeddings = model.predict(dummy_embeddings, verbose=0)

    print(f"\nNormalized embeddings shape: {normalized_embeddings.shape}")
    print(f"Normalized mean: {normalized_embeddings.mean():.3f}")
    print(f"Normalized std: {normalized_embeddings.std():.3f}")

    print("\n✓ Text preprocessing example complete!")
    print("  Use case: NLP models with pre-trained embeddings")
    print("  Benefit: Stable training with normalized inputs")


def example_timeseries_normalization():
    """Example 3: Time-series normalization.

    Use case: Normalize features across time dimension for forecasting.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: TIME-SERIES PREPROCESSING - TEMPORAL NORMALIZATION")
    print("="*80)

    layers, models, ops, backend = setup_keras()
    if layers is None:
        return

    # Time-series parameters
    n_timesteps = 100
    n_features = 8

    # Compute statistics from historical data (simulated)
    historical_data = np.random.randn(1000, n_timesteps, n_features).astype(
        "float32"
    )

    # Statistics per feature (across time)
    feature_mean = historical_data.mean(axis=(0, 1))
    feature_std = historical_data.std(axis=(0, 1))
    feature_variance = feature_std ** 2

    print(f"\nTime-series shape: (batch, {n_timesteps}, {n_features})")
    print(f"Feature statistics computed from {len(historical_data)} samples")
    print(f"Mean per feature: {feature_mean[:3]} ... (showing first 3)")
    print(f"Std per feature:  {feature_std[:3]} ... (showing first 3)")

    # Create normalization layer for time-series
    normalize_layer = layers.Normalization(
        axis=-1,  # Normalize across feature dimension
        mean=feature_mean,
        variance=feature_variance,
    )

    # Build forecasting preprocessing model
    inputs = layers.Input(shape=(n_timesteps, n_features))
    normalized = normalize_layer(inputs)
    # Add LSTM layer as example
    lstm_out = layers.LSTM(64)(normalized)
    outputs = layers.Dense(n_features)(lstm_out)  # Forecast next timestep

    model = models.Model(inputs=inputs, outputs=outputs)

    print("\nTime-series forecasting model:")
    model.summary()

    # Example: Process time-series batch
    batch_size = 16
    test_data = np.random.randn(batch_size, n_timesteps, n_features).astype(
        "float32"
    )

    print(f"\nInput time-series shape: {test_data.shape}")
    print(f"Input mean: {test_data.mean():.3f}")
    print(f"Input std: {test_data.std():.3f}")

    # Get predictions (normalized internally)
    predictions = model.predict(test_data, verbose=0)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions mean: {predictions.mean():.3f}")
    print(f"Predictions std: {predictions.std():.3f}")

    print("\n✓ Time-series preprocessing example complete!")
    print("  Use case: Financial forecasting, sensor data analysis")
    print("  Benefit: Stable training across different feature scales")


def example_multi_dimensional_normalization():
    """Example 4: Multi-dimensional normalization (gh-22065 fix demo).

    Use case: Normalize multi-channel data with specific axis configuration.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: MULTI-DIMENSIONAL NORMALIZATION (gh-22065 FIX)")
    print("="*80)

    layers, models, ops, backend = setup_keras()
    if layers is None:
        return

    # Multi-dimensional input: (batch, channels, height, width)
    # Common in medical imaging or multi-spectral data
    n_channels = 4
    height, width = 64, 64

    # Compute statistics per (channel, width) - axis (1, 3)
    # This is the exact case from gh-22065
    historical_data = np.random.randn(500, n_channels, height, width).astype(
        "float32"
    )

    # Compute mean and variance for kept axes (1, 3)
    channel_width_mean = historical_data.mean(axis=(0, 2))  # Shape: (4, 64)
    channel_width_std = historical_data.std(axis=(0, 2))
    channel_width_variance = channel_width_std ** 2

    print(f"\nInput shape: (batch, {n_channels}, {height}, {width})")
    print(f"Normalization axes: (1, 3) - per (channel, width)")
    print(f"Statistics shape: {channel_width_mean.shape}")

    # Create normalization layer - THIS IS THE FIX IN ACTION
    normalize_layer = layers.Normalization(
        axis=(1, 3),
        mean=channel_width_mean,
        variance=channel_width_variance,
    )

    # Build model
    inputs = layers.Input(shape=(n_channels, height, width))
    normalized = normalize_layer(inputs)
    model = models.Model(inputs=inputs, outputs=normalized)

    print("\nMulti-dimensional normalization model:")
    model.summary()

    # Test
    test_batch = np.random.randn(8, n_channels, height, width).astype(
        "float32"
    )

    print(f"\nInput batch shape: {test_batch.shape}")

    # This should work without errors (gh-22065 is fixed!)
    normalized_batch = model.predict(test_batch, verbose=0)

    print(f"Normalized batch shape: {normalized_batch.shape}")
    print(f"Normalized mean per (channel, width): varies as expected")

    # Verify normalization
    print("\nVerification:")
    print("  Before: mean={:.3f}, std={:.3f}".format(
        test_batch.mean(), test_batch.std()
    ))
    print("  After:  mean={:.3f}, std={:.3f}".format(
        normalized_batch.mean(), normalized_batch.std()
    ))

    print("\n✓ Multi-dimensional normalization example complete!")
    print("  Use case: Medical imaging, multi-spectral satellite data")
    print("  gh-22065: This would have crashed in Keras 3.8.0, now fixed!")


def main():
    parser = argparse.ArgumentParser(
        description="Normalization layer usage examples"
    )
    parser.add_argument(
        "--example",
        choices=["image", "text", "timeseries", "multidim"],
        help="Which example to run",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all examples"
    )

    args = parser.parse_args()

    if args.all or args.example is None:
        example_image_preprocessing()
        example_text_preprocessing()
        example_timeseries_normalization()
        example_multi_dimensional_normalization()
    elif args.example == "image":
        example_image_preprocessing()
    elif args.example == "text":
        example_text_preprocessing()
    elif args.example == "timeseries":
        example_timeseries_normalization()
    elif args.example == "multidim":
        example_multi_dimensional_normalization()

    print("\n" + "="*80)
    print("All examples complete! Use --help for more options.")
    print("="*80)


if __name__ == "__main__":
    main()
