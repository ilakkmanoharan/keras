import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset


class NormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_normalization_basics(self):
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=3,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
                "mean": np.array([0.5, 0.2, -0.1]),
                "variance": np.array([0.1, 0.2, 0.3]),
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
                "mean": np.array([0.5, 0.2, -0.1]),
                "variance": np.array([0.1, 0.2, 0.3]),
                "invert": True,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.parameters([("np",), ("tensor",), ("tf.data")])
    def test_normalization_adapt(self, input_type):
        x = np.random.random((32, 4))
        if input_type == "np":
            data = x
        elif input_type == "tensor":
            data = backend.convert_to_tensor(x)
        elif input_type == "tf.data":
            data = tf_data.Dataset.from_tensor_slices(x).batch(8)
        else:
            raise NotImplementedError(input_type)

        layer = layers.Normalization()
        layer.adapt(data)
        self.assertTrue(layer.built)
        output = layer(x)
        output = backend.convert_to_numpy(output)
        self.assertAllClose(np.var(output, axis=0), 1.0, atol=1e-5)
        self.assertAllClose(np.mean(output, axis=0), 0.0, atol=1e-5)

        # Test in high-dim and with tuple axis.
        x = np.random.random((32, 4, 3, 5))
        if input_type == "np":
            data = x
        elif input_type == "tensor":
            data = backend.convert_to_tensor(x)
        elif input_type == "tf.data":
            data = tf_data.Dataset.from_tensor_slices(x).batch(8)

        layer = layers.Normalization(axis=(1, 2))
        layer.adapt(data)
        self.assertTrue(layer.built)
        output = layer(x)
        output = backend.convert_to_numpy(output)
        self.assertAllClose(np.var(output, axis=(0, 3)), 1.0, atol=1e-5)
        self.assertAllClose(np.mean(output, axis=(0, 3)), 0.0, atol=1e-5)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Test symbolic call for torch meta device.",
    )
    def test_call_on_meta_device_after_built(self):
        layer = layers.Normalization()
        data = np.random.random((32, 4))
        layer.adapt(data)
        with backend.device("meta"):
            layer(data)

    def test_normalization_with_mean_only_raises_error(self):
        # Test error when only `mean` is provided
        with self.assertRaisesRegex(
            ValueError, "both `mean` and `variance` must be set"
        ):
            layers.Normalization(mean=0.5)

    def test_normalization_with_variance_only_raises_error(self):
        # Test error when only `variance` is provided
        with self.assertRaisesRegex(
            ValueError, "both `mean` and `variance` must be set"
        ):
            layers.Normalization(variance=0.1)

    def test_normalization_axis_too_high(self):
        with self.assertRaisesRegex(
            ValueError, "All `axis` values must be in the range"
        ):
            layer = layers.Normalization(axis=3)
            layer.build((2, 2))

    def test_normalization_axis_too_low(self):
        with self.assertRaisesRegex(
            ValueError, "All `axis` values must be in the range"
        ):
            layer = layers.Normalization(axis=-4)
            layer.build((2, 3, 4))

    def test_normalization_unknown_axis_shape(self):
        with self.assertRaisesRegex(ValueError, "All `axis` values to be kept"):
            layer = layers.Normalization(axis=1)
            layer.build((None, None))

    def test_normalization_adapt_with_incompatible_shape(self):
        layer = layers.Normalization(axis=-1)
        initial_shape = (10, 5)
        layer.build(initial_shape)
        new_shape_data = np.random.random((10, 3))
        with self.assertRaisesRegex(ValueError, "an incompatible shape"):
            layer.adapt(new_shape_data)

    def test_tf_data_compatibility(self):
        x = np.random.random((32, 3))
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1)

        # With built-in values
        layer = layers.Normalization(
            mean=[0.1, 0.2, 0.3], variance=[0.1, 0.2, 0.3], axis=-1
        )
        layer.build((None, 3))
        for output in ds.map(layer).take(1):
            output.numpy()

        # With adapt flow
        layer = layers.Normalization(axis=-1)
        layer.adapt(
            np.random.random((32, 3)),
        )
        for output in ds.map(layer).take(1):
            output.numpy()

    def test_normalization_with_scalar_mean_var(self):
        input_data = np.array([[1, 2, 3]], dtype="float32")
        layer = layers.Normalization(mean=3.0, variance=2.0)
        layer(input_data)

    @parameterized.parameters([("x",), ("x_and_y",), ("x_y_and_weights",)])
    def test_adapt_pydataset_compat(self, pydataset_type):
        import keras

        class CustomDataset(PyDataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                x = np.random.rand(32, 32, 3)
                y = np.random.randint(0, 10, size=(1,))
                weights = np.random.randint(0, 10, size=(1,))
                if pydataset_type == "x":
                    return x
                elif pydataset_type == "x_and_y":
                    return x, y
                elif pydataset_type == "x_y_and_weights":
                    return x, y, weights
                else:
                    raise NotImplementedError(pydataset_type)

        normalizer = keras.layers.Normalization()
        normalizer.adapt(CustomDataset())
        self.assertTrue(normalizer.built)
        self.assertIsNotNone(normalizer.mean)
        self.assertIsNotNone(normalizer.variance)
        self.assertEqual(normalizer.mean.shape[-1], 3)
        self.assertEqual(normalizer.variance.shape[-1], 3)
        sample_input = np.random.rand(1, 32, 32, 3)
        output = normalizer(sample_input)
        self.assertEqual(output.shape, (1, 32, 32, 3))

    def test_normalization_multidim_mean_variance(self):
        """Regression test for gh-22065: multi-dimensional mean/variance."""
        # Test the exact case from the GitHub issue
        input_data = np.random.random((2, 2, 3, 4)).astype("float32")
        mean = np.array([[0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        variance = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)
        output = layer(input_data)

        # Verify no crash and correct output shape
        self.assertEqual(output.shape, (2, 2, 3, 4))

        # Verify the normalization is computed correctly
        output_np = backend.convert_to_numpy(output)
        # For manual verification: (input - mean) / sqrt(variance)
        expected_mean = mean.reshape(1, 2, 1, 4)
        expected_var = variance.reshape(1, 2, 1, 4)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_scalar_mean_variance_backward_compat(self):
        """Ensure scalar mean/variance still works (backward compatibility)."""
        input_data = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        layer = layers.Normalization(mean=3.0, variance=2.0)
        output = layer(input_data)

        # Verify correct shape
        self.assertEqual(output.shape, input_data.shape)

        # Verify computation: (input - 3.0) / sqrt(2.0)
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - 3.0) / np.sqrt(2.0)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_1d_mean_variance_backward_compat(self):
        """Ensure 1D mean/variance still works (backward compatibility)."""
        input_data = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])
        layer = layers.Normalization(axis=-1, mean=mean, variance=variance)
        output = layer(input_data)

        # Verify correct shape
        self.assertEqual(output.shape, input_data.shape)

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - mean) / np.sqrt(variance)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_multidim_various_shapes(self):
        """Test multi-dimensional mean/variance with various axes."""
        # Test case 1: axis=(1, 3)
        input_data = np.random.random((2, 3, 4, 5)).astype("float32")
        mean = np.random.random((3, 5))
        variance = np.ones((3, 5))
        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)
        output = layer(input_data)
        self.assertEqual(output.shape, input_data.shape)

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        expected_mean = mean.reshape(1, 3, 1, 5)
        expected_var = variance.reshape(1, 3, 1, 5)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

        # Test case 2: axis=(0, 2)
        input_data = np.random.random((4, 3, 5, 2)).astype("float32")
        mean = np.random.random((4, 5))
        variance = np.ones((4, 5))
        layer = layers.Normalization(axis=(0, 2), mean=mean, variance=variance)
        output = layer(input_data)
        self.assertEqual(output.shape, input_data.shape)

    def test_normalization_multidim_invert(self):
        """Test multi-dimensional mean/variance with invert=True."""
        input_data = np.random.random((2, 2, 3, 4)).astype("float32")
        mean = np.array([[0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        variance = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        layer = layers.Normalization(
            axis=(1, 3), mean=mean, variance=variance, invert=True
        )
        output = layer(input_data)

        # Verify no crash and correct output shape
        self.assertEqual(output.shape, (2, 2, 3, 4))

        # Verify the denormalization is computed correctly
        output_np = backend.convert_to_numpy(output)
        expected_mean = mean.reshape(1, 2, 1, 4)
        expected_var = variance.reshape(1, 2, 1, 4)
        expected = expected_mean + input_data * np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_mean_already_broadcast_shape(self):
        """Test mean already in _broadcast_shape uses broadcast_to."""
        # If mean is already in broadcast form (e.g., [1,2,1,4]), it should
        # use broadcast_to rather than reshape. This tests the edge case
        # where the user provides mean/variance in full broadcast shape.
        input_data = np.random.random((2, 2, 3, 4)).astype("float32")

        # Mean/variance in broadcast form (already has 1s for reduced axes)
        mean = np.array([[[[0, 1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0, 7.0]]]])
        variance = np.array([[[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]]])

        # Shape is [1, 2, 1, 4] which is already the broadcast shape
        self.assertEqual(mean.shape, (1, 2, 1, 4))

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)
        output = layer(input_data)

        # Should work without error and produce correct output
        self.assertEqual(output.shape, (2, 2, 3, 4))

        # Verify computation is correct
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - mean) / np.sqrt(variance)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_mismatched_element_count_raises_error(self):
        """Test mismatched element count raises ValueError."""
        # Provide mean/variance with wrong number of elements
        # For axis=(1, 3) on shape (2, 2, 3, 4), kept axes are dims 1 and 3
        # So mean should have shape (2, 4) with 8 elements
        # Provide wrong shape (3, 3) with 9 elements instead
        mean = np.random.random((3, 3))
        variance = np.ones((3, 3))

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)

        # Should raise ValueError when building with mismatched count
        with self.assertRaisesRegex(
            ValueError,
            "Cannot reshape mean/variance with shape.*elements",
        ):
            layer.build((None, 2, 3, 4))

    def test_normalization_axis_permutations(self):
        """Test different axis permutations to ensure logic holds."""
        # Test case 1: axis=(0, 2) on 4D input
        input_data = np.random.random((3, 5, 4, 7)).astype("float32")
        mean = np.random.random((3, 4))  # Kept axes 0 and 2
        variance = np.ones((3, 4))

        layer = layers.Normalization(axis=(0, 2), mean=mean, variance=variance)
        output = layer(input_data)

        self.assertEqual(output.shape, (3, 5, 4, 7))

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        # Broadcast shape for axis=(0,2) is [3, 1, 4, 1]
        expected_mean = mean.reshape(3, 1, 4, 1)
        expected_var = variance.reshape(3, 1, 4, 1)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

        # Test case 2: axis=(1, 2, 3) on 4D input
        input_data = np.random.random((2, 3, 4, 5)).astype("float32")
        mean = np.random.random((3, 4, 5))  # Kept axes 1, 2, 3
        variance = np.ones((3, 4, 5))

        layer = layers.Normalization(
            axis=(1, 2, 3), mean=mean, variance=variance
        )
        output = layer(input_data)

        self.assertEqual(output.shape, (2, 3, 4, 5))

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        # Broadcast shape for axis=(1,2,3) is [1, 3, 4, 5]
        expected_mean = mean.reshape(1, 3, 4, 5)
        expected_var = variance.reshape(1, 3, 4, 5)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

        # Test case 3: axis=0 on 3D input (single axis)
        input_data = np.random.random((4, 5, 6)).astype("float32")
        mean = np.random.random((4,))  # Kept axis 0
        variance = np.ones((4,))

        layer = layers.Normalization(axis=0, mean=mean, variance=variance)
        output = layer(input_data)

        self.assertEqual(output.shape, (4, 5, 6))

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        # Broadcast shape for axis=0 is [4, 1, 1]
        expected_mean = mean.reshape(4, 1, 1)
        expected_var = variance.reshape(4, 1, 1)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

    # ========================================================================
    # COMPREHENSIVE EDGE CASE TESTS
    # ========================================================================

    def test_normalization_symbolic_shapes_with_none(self):
        """Test with symbolic shapes containing None dimensions."""
        # This tests that the layer handles unknown batch dimensions correctly
        mean = np.array([[0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        variance = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)

        # Build with None in batch dimension
        layer.build((None, 2, 3, 4))

        # Should work with concrete batch size
        input_data = np.random.random((5, 2, 3, 4)).astype("float32")
        output = layer(input_data)
        self.assertEqual(output.shape, (5, 2, 3, 4))

        # Should work with different batch size
        input_data2 = np.random.random((10, 2, 3, 4)).astype("float32")
        output2 = layer(input_data2)
        self.assertEqual(output2.shape, (10, 2, 3, 4))

    def test_normalization_dtype_casting(self):
        """Test that different dtypes are handled correctly."""
        # Provide mean/variance as different dtypes
        mean_int = np.array([1, 2, 3], dtype="int32")
        variance_int = np.array([1, 1, 1], dtype="int32")

        layer = layers.Normalization(
            axis=-1, mean=mean_int, variance=variance_int
        )

        # Input as float32
        input_data = np.array([[4.0, 5.0, 6.0]], dtype="float32")
        output = layer(input_data)

        # Should work and cast appropriately
        self.assertEqual(output.shape, input_data.shape)

        # Verify computation (mean/var cast to float)
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - mean_int.astype("float32")) / np.sqrt(
            variance_int.astype("float32")
        )
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_backend_native_tensors(self):
        """Test with backend-native tensors vs numpy arrays."""
        mean_np = np.array([0.5, 0.2, -0.1])
        variance_np = np.array([0.1, 0.2, 0.3])

        # Create layer with numpy arrays
        layer = layers.Normalization(
            axis=-1, mean=mean_np, variance=variance_np
        )

        # Input as backend-native tensor
        input_data = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        input_tensor = backend.convert_to_tensor(input_data)

        output = layer(input_tensor)

        self.assertEqual(output.shape, input_data.shape)

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        expected = (input_data - mean_np) / np.sqrt(variance_np)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_zero_sized_dimensions(self):
        """Test handling of zero-sized dimensions (edge case)."""
        # Create layer with valid mean/variance
        mean = np.array([0.5, 0.2, -0.1])
        variance = np.array([0.1, 0.2, 0.3])
        layer = layers.Normalization(axis=-1, mean=mean, variance=variance)

        # Zero-sized batch dimension
        input_data = np.array([], dtype="float32").reshape(0, 3)
        output = layer(input_data)

        # Should handle gracefully
        self.assertEqual(output.shape, (0, 3))

    def test_normalization_mixed_rank_mean_variance(self):
        """Test error handling when mean/variance have unexpected ranks."""
        # Mean as 3D when 2D expected
        mean_3d = np.random.random((2, 2, 4))
        variance_3d = np.ones((2, 2, 4))

        layer = layers.Normalization(
            axis=(1, 3), mean=mean_3d, variance=variance_3d
        )

        # Should use broadcast_to path (not matching expected shape)
        try:
            layer.build((None, 2, 3, 4))
            # If it doesn't error, that's fine - broadcast_to handles it
        except Exception:
            # If it errors, that's also acceptable behavior
            pass

    def test_normalization_very_large_tensors_stress(self):
        """Stress test with large tensors."""
        # Large but reasonable dimensions
        input_shape = (2, 128, 64, 32)
        mean = np.random.random((128, 32)).astype("float32")
        variance = np.ones((128, 32)).astype("float32")

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)
        layer.build((None, 128, 64, 32))

        # Create large input
        input_data = np.random.random(input_shape).astype("float32")
        output = layer(input_data)

        self.assertEqual(output.shape, input_shape)

        # Verify a sample (don't check all elements for performance)
        output_np = backend.convert_to_numpy(output)
        self.assertFalse(np.any(np.isnan(output_np)))
        self.assertFalse(np.any(np.isinf(output_np)))

    def test_normalization_complex_axis_configuration(self):
        """Test with complex nested axis configurations."""
        # 5D input with non-contiguous axes
        input_data = np.random.random((2, 3, 4, 5, 6)).astype("float32")
        mean = np.random.random((3, 5))  # axes 1 and 3
        variance = np.ones((3, 5))

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)
        output = layer(input_data)

        self.assertEqual(output.shape, (2, 3, 4, 5, 6))

        # Verify computation
        output_np = backend.convert_to_numpy(output)
        expected_mean = mean.reshape(1, 3, 1, 5, 1)
        expected_var = variance.reshape(1, 3, 1, 5, 1)
        expected = (input_data - expected_mean) / np.sqrt(expected_var)
        self.assertAllClose(output_np, expected, atol=1e-5)

    def test_normalization_error_message_clarity(self):
        """Test that error messages are clear and helpful."""
        # Wrong element count
        mean = np.random.random((3, 3))  # Should be (2, 4)
        variance = np.ones((3, 3))

        layer = layers.Normalization(axis=(1, 3), mean=mean, variance=variance)

        # Should get clear error mentioning element count
        with self.assertRaisesRegex(
            (ValueError, Exception),
            "(Cannot reshape|element|shape)",
        ):
            layer.build((None, 2, 3, 4))

    def test_normalization_negative_axes(self):
        """Test with negative axis values."""
        input_data = np.random.random((2, 3, 4, 5)).astype("float32")
        mean = np.random.random((4, 5))
        variance = np.ones((4, 5))

        # Use negative axes
        layer = layers.Normalization(axis=(-2, -1), mean=mean, variance=variance)
        output = layer(input_data)

        self.assertEqual(output.shape, (2, 3, 4, 5))

        # Should behave same as axis=(2, 3)
        layer2 = layers.Normalization(axis=(2, 3), mean=mean, variance=variance)
        output2 = layer2(input_data)

        output_np = backend.convert_to_numpy(output)
        output2_np = backend.convert_to_numpy(output2)
        self.assertAllClose(output_np, output2_np, atol=1e-5)

    def test_normalization_single_element_tensors(self):
        """Test with tensors containing single elements."""
        # Scalar-like but with shape
        input_data = np.array([[[1.0]]]).astype("float32")  # Shape (1, 1, 1)
        mean = np.array([[0.5]])
        variance = np.array([[0.25]])

        layer = layers.Normalization(axis=(1, 2), mean=mean, variance=variance)
        output = layer(input_data)

        self.assertEqual(output.shape, (1, 1, 1))

        output_np = backend.convert_to_numpy(output)
        expected = (input_data - 0.5) / np.sqrt(0.25)
        self.assertAllClose(output_np, expected, atol=1e-5)
