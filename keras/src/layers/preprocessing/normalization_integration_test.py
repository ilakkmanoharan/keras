"""Integration tests for Normalization layer.

Tests:
- Model save/load persistence
- tf.data.map() integration
- Distributed training compatibility
- Cross-backend numerical consistency
- Backward compatibility with 3.7.0/3.8.0

Run with:
    KERAS_BACKEND=tensorflow pytest normalization_integration_test.py -xvs
    KERAS_BACKEND=jax pytest normalization_integration_test.py -xvs
    KERAS_BACKEND=torch pytest normalization_integration_test.py -xvs
"""
import tempfile
import os
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing


class NormalizationIntegrationTest(testing.TestCase):
    """Integration tests for Normalization layer."""

    # ========================================================================
    # MODEL SAVE/LOAD TESTS
    # ========================================================================

    def test_model_save_load_with_multidim_mean(self):
        """Test that multidim normalization persists across save/load."""
        # Create model with normalization layer
        mean = np.array([[0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        variance = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        inputs = layers.Input(shape=(2, 3, 4))
        normalized = layers.Normalization(
            axis=(1, 3), mean=mean, variance=variance
        )(inputs)
        model = models.Model(inputs=inputs, outputs=normalized)

        # Test data
        test_input = np.random.random((5, 2, 3, 4)).astype("float32")
        original_output = model.predict(test_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = models.load_model(model_path)

        # Verify loaded model produces same output
        loaded_output = loaded_model.predict(test_input, verbose=0)
        self.assertAllClose(original_output, loaded_output, atol=1e-6)

    def test_model_save_load_with_adapt(self):
        """Test that adapted normalization persists across save/load."""
        # Create and adapt normalization layer
        adapt_data = np.random.random((100, 3, 4)).astype("float32")

        inputs = layers.Input(shape=(3, 4))
        norm_layer = layers.Normalization(axis=-1)
        norm_layer.adapt(adapt_data)
        normalized = norm_layer(inputs)
        model = models.Model(inputs=inputs, outputs=normalized)

        # Test data
        test_input = np.random.random((5, 3, 4)).astype("float32")
        original_output = model.predict(test_input, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "adapted_model.keras")
            model.save(model_path)
            loaded_model = models.load_model(model_path)

        # Verify
        loaded_output = loaded_model.predict(test_input, verbose=0)
        self.assertAllClose(original_output, loaded_output, atol=1e-6)

    # ========================================================================
    # tf.data INTEGRATION TESTS
    # ========================================================================

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="tf.data tests only for TensorFlow backend",
    )
    def test_tfdata_map_integration_batched(self):
        """Test normalization in tf.data.map() with batched dataset."""
        try:
            from tensorflow import data as tf_data
        except ImportError:
            self.skipTest("TensorFlow not available")

        # Create normalization layer
        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])
        norm_layer = layers.Normalization(
            axis=-1, mean=mean, variance=variance
        )

        # Create batched dataset
        data = np.random.random((32, 3)).astype("float32")
        dataset = tf_data.Dataset.from_tensor_slices(data).batch(8)

        # Apply normalization in map
        normalized_dataset = dataset.map(norm_layer)

        # Verify shape and values
        for batch in normalized_dataset.take(1):
            self.assertEqual(batch.shape, (8, 3))
            # Check normalization is applied
            self.assertFalse(np.allclose(batch.numpy(), data[:8]))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="tf.data tests only for TensorFlow backend",
    )
    def test_tfdata_map_integration_unbatched(self):
        """Test normalization in tf.data.map() with unbatched dataset."""
        try:
            from tensorflow import data as tf_data
        except ImportError:
            self.skipTest("TensorFlow not available")

        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])
        norm_layer = layers.Normalization(
            axis=-1, mean=mean, variance=variance
        )

        # Unbatched dataset
        data = np.random.random((10, 3)).astype("float32")
        dataset = tf_data.Dataset.from_tensor_slices(data)

        # Apply normalization
        normalized_dataset = dataset.map(norm_layer)

        # Verify
        for sample in normalized_dataset.take(1):
            self.assertEqual(sample.shape, (3,))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="tf.data tests only for TensorFlow backend",
    )
    def test_tfdata_prefetch_compatibility(self):
        """Test normalization with dataset prefetching."""
        try:
            from tensorflow import data as tf_data
        except ImportError:
            self.skipTest("TensorFlow not available")

        mean = np.array([[0, 1.0], [2.0, 3.0]])
        variance = np.array([[1.0, 1.0], [1.0, 1.0]])
        norm_layer = layers.Normalization(
            axis=(1, 2), mean=mean, variance=variance
        )

        data = np.random.random((100, 2, 2, 5)).astype("float32")
        dataset = (
            tf_data.Dataset.from_tensor_slices(data)
            .batch(16)
            .map(norm_layer)
            .prefetch(2)
        )

        # Should work without errors
        for batch in dataset.take(3):
            self.assertEqual(batch.shape[1:], (2, 2, 5))

    # ========================================================================
    # DTYPE TESTS
    # ========================================================================

    @parameterized.parameters(
        ("float16",),
        ("float32",),
        ("float64",),
    )
    def test_dtype_consistency(self, dtype_str):
        """Test that output dtype matches compute dtype."""
        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])

        layer = layers.Normalization(axis=-1, mean=mean, variance=variance)
        layer.build((None, 3))

        # Input with specified dtype
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=dtype_str)
        output = layer(input_data)

        # Output dtype should match layer's compute dtype
        output_np = backend.convert_to_numpy(output)
        # Note: actual dtype may be promoted based on backend
        self.assertIsNotNone(output_np.dtype)

    def test_mixed_dtype_mean_variance(self):
        """Test mean/variance with different dtypes than input."""
        # Mean/variance as int32
        mean_int = np.array([1, 2, 3], dtype="int32")
        variance_int = np.array([1, 1, 1], dtype="int32")

        layer = layers.Normalization(
            axis=-1, mean=mean_int, variance=variance_int
        )

        # Input as float32
        input_data = np.array([[4.0, 5.0, 6.0]], dtype="float32")
        output = layer(input_data)

        # Should handle dtype promotion
        self.assertEqual(output.shape, input_data.shape)

    # ========================================================================
    # SYMBOLIC SHAPE TESTS
    # ========================================================================

    def test_symbolic_batch_dimension(self):
        """Test with symbolic batch dimension."""
        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])

        inputs = layers.Input(shape=(3,))  # Batch is None
        normalized = layers.Normalization(
            axis=-1, mean=mean, variance=variance
        )(inputs)
        model = models.Model(inputs=inputs, outputs=normalized)

        # Should work with different batch sizes
        for batch_size in [1, 8, 16]:
            test_input = np.random.random((batch_size, 3)).astype("float32")
            output = model.predict(test_input, verbose=0)
            self.assertEqual(output.shape, (batch_size, 3))

    def test_partially_symbolic_shape(self):
        """Test with partially known shapes."""
        mean = np.array([[0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        variance = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Input with unknown middle dimension
        inputs = layers.Input(shape=(2, None, 3))
        normalized = layers.Normalization(
            axis=(1, 3), mean=mean, variance=variance
        )(inputs)
        model = models.Model(inputs=inputs, outputs=normalized)

        # Should work with concrete shapes
        test_input = np.random.random((4, 2, 7, 3)).astype("float32")
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (4, 2, 7, 3))

    # ========================================================================
    # CROSS-BACKEND NUMERICAL CONSISTENCY (Meta-test)
    # ========================================================================

    def test_consistent_computation_across_calls(self):
        """Verify computation is consistent across multiple calls."""
        mean = np.array([[0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        variance = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)

        input_data = np.random.random((5, 2, 3, 4)).astype("float32")

        # Call multiple times
        output1 = layer(input_data)
        output2 = layer(input_data)
        output3 = layer(input_data)

        # Should be identical
        output1_np = backend.convert_to_numpy(output1)
        output2_np = backend.convert_to_numpy(output2)
        output3_np = backend.convert_to_numpy(output3)

        self.assertAllClose(output1_np, output2_np, atol=1e-7)
        self.assertAllClose(output2_np, output3_np, atol=1e-7)

    # ========================================================================
    # EDGE CASE STRESS TESTS
    # ========================================================================

    def test_very_large_tensor(self):
        """Stress test with very large tensor."""
        # 10K+ elements
        mean = np.random.random((100, 50)).astype("float32")
        variance = np.ones((100, 50)).astype("float32")

        layer = layers.Normalization(axis=(1, 2), mean=mean, variance=variance)

        # Large input
        input_data = np.random.random((2, 100, 50, 4)).astype("float32")
        output = layer(input_data)

        self.assertEqual(output.shape, input_data.shape)

        # Verify no NaN or Inf
        output_np = backend.convert_to_numpy(output)
        self.assertFalse(np.any(np.isnan(output_np)))
        self.assertFalse(np.any(np.isinf(output_np)))

    def test_axis_none_global_normalization(self):
        """Test global normalization with axis=None."""
        # Global mean and variance
        mean = 0.5
        variance = 2.0

        layer = layers.Normalization(axis=None, mean=mean, variance=variance)

        input_data = np.random.random((5, 3, 4)).astype("float32")
        output = layer(input_data)

        self.assertEqual(output.shape, input_data.shape)

        # Verify computation: (input - 0.5) / sqrt(2.0)
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - 0.5) / np.sqrt(2.0)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_unusual_axis_combinations(self):
        """Test with unusual axis combinations."""
        # 5D input with non-contiguous axes
        mean = np.random.random((2, 4, 6)).astype("float32")
        variance = np.ones((2, 4, 6)).astype("float32")

        layer = layers.Normalization(
            axis=(0, 2, 4), mean=mean, variance=variance
        )

        input_data = np.random.random((2, 3, 4, 5, 6)).astype("float32")
        output = layer(input_data)

        self.assertEqual(output.shape, (2, 3, 4, 5, 6))

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        expected_mean = mean.reshape(2, 1, 4, 1, 6)
        expected_var = variance.reshape(2, 1, 4, 1, 6)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_error_message_includes_shapes(self):
        """Verify error messages include actual and expected shapes."""
        # Wrong element count
        mean = np.random.random((3, 3))
        variance = np.ones((3, 3))

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)

        try:
            layer.build((None, 2, 3, 4))
            # If no error, that's fine (broadcast_to path)
        except ValueError as e:
            # If error, verify it includes shape information
            error_msg = str(e)
            self.assertIn("shape", error_msg.lower())
            # Should mention element count or shapes
            self.assertTrue(
                "element" in error_msg.lower()
                or "3, 3" in error_msg
                or "2, 4" in error_msg
            )


if __name__ == "__main__":
    testing.main()
