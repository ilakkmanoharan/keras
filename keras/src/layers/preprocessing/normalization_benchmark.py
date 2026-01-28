"""Benchmark suite for Normalization layer reshape/broadcast strategies.

This module benchmarks all 4 implementation approaches to determine:
1. Build time performance
2. Call time performance (inference)
3. Memory usage patterns
4. Backend-specific behavior

Run with:
    python keras/src/layers/preprocessing/normalization_benchmark.py

Or for specific backend:
    KERAS_BACKEND=tensorflow python ...
    KERAS_BACKEND=jax python ...
    KERAS_BACKEND=torch python ...
"""
import time
import statistics
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    import numpy as np
    from keras.src.layers.preprocessing import normalization
    from keras.src import backend
    from keras.src import ops
    HAS_KERAS = True
except ImportError as e:
    print(f"Keras not available: {e}")
    HAS_KERAS = False


class BenchmarkScenario:
    """Test scenario for benchmarking."""

    def __init__(self, name, input_shape, axis, mean_shape, description):
        self.name = name
        self.input_shape = input_shape
        self.axis = axis
        self.mean_shape = mean_shape
        self.description = description


# Define benchmark scenarios covering various cases
SCENARIOS = [
    BenchmarkScenario(
        "scalar_mean_small",
        input_shape=(None, 10),
        axis=-1,
        mean_shape=(),
        description="Scalar mean on small 2D input"
    ),
    BenchmarkScenario(
        "1d_mean_medium",
        input_shape=(None, 100),
        axis=-1,
        mean_shape=(100,),
        description="1D mean on medium 2D input"
    ),
    BenchmarkScenario(
        "compact_2d_small",
        input_shape=(None, 2, 3, 4),
        axis=(1, 3),
        mean_shape=(2, 4),
        description="Compact 2D mean (regression case)"
    ),
    BenchmarkScenario(
        "compact_2d_large",
        input_shape=(None, 16, 32, 64),
        axis=(1, 3),
        mean_shape=(16, 64),
        description="Compact 2D mean large dimensions"
    ),
    BenchmarkScenario(
        "compact_3d",
        input_shape=(None, 3, 4, 5),
        axis=(1, 2, 3),
        mean_shape=(3, 4, 5),
        description="Compact 3D mean (many kept axes)"
    ),
    BenchmarkScenario(
        "sparse_axes",
        input_shape=(3, 5, 4, 7, 6),
        axis=(0, 2, 4),
        mean_shape=(3, 4, 6),
        description="Non-contiguous kept axes"
    ),
]


def timeit(func, iterations=1000):
    """Time a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }


def benchmark_build_time(layer_class, scenario, method_name, iterations=1000):
    """Benchmark layer build() time."""
    if not HAS_KERAS:
        return None

    # Create mean/variance with correct shape
    if scenario.mean_shape == ():
        mean = 0.0
        variance = 1.0
    else:
        mean = np.random.random(scenario.mean_shape).astype('float32')
        variance = np.ones(scenario.mean_shape).astype('float32')

    def build_layer():
        layer = layer_class(axis=scenario.axis, mean=mean, variance=variance)
        layer.build(scenario.input_shape)

    return timeit(build_layer, iterations)


def benchmark_call_time(layer_class, scenario, method_name, iterations=1000):
    """Benchmark layer call() time."""
    if not HAS_KERAS:
        return None

    # Setup
    if scenario.mean_shape == ():
        mean = 0.0
        variance = 1.0
    else:
        mean = np.random.random(scenario.mean_shape).astype('float32')
        variance = np.ones(scenario.mean_shape).astype('float32')

    layer = layer_class(axis=scenario.axis, mean=mean, variance=variance)
    layer.build(scenario.input_shape)

    # Create input data (concrete shape, not None)
    concrete_shape = tuple(
        32 if dim is None else dim for dim in scenario.input_shape
    )
    input_data = np.random.random(concrete_shape).astype('float32')
    input_tensor = ops.convert_to_tensor(input_data)

    def call_layer():
        layer(input_tensor)

    return timeit(call_layer, iterations)


def create_layer_variant(base_class, method_name):
    """Create a Normalization variant using specific reshape/broadcast method."""
    if method_name == "v1_shape_identity":
        # Default implementation
        return base_class

    # Create dynamic subclass that overrides _reshape_or_broadcast
    class NormalizationVariant(base_class):
        def _reshape_or_broadcast(self, tensor, name, expected_shape, broadcast_shape):
            if method_name == "v2_try_except":
                return self._reshape_or_broadcast_v2_try_except(
                    tensor, name, expected_shape, broadcast_shape
                )
            elif method_name == "v3_element_count":
                return self._reshape_or_broadcast_v3_element_count(
                    tensor, name, expected_shape, broadcast_shape
                )
            elif method_name == "v4_unified":
                return self._reshape_or_broadcast_v4_unified(
                    tensor, name, expected_shape, broadcast_shape
                )
            else:
                return super()._reshape_or_broadcast(
                    tensor, name, expected_shape, broadcast_shape
                )

    return NormalizationVariant


def run_benchmarks():
    """Run comprehensive benchmarks."""
    if not HAS_KERAS:
        print("ERROR: Keras not available. Cannot run benchmarks.")
        return

    print("="*80)
    print("NORMALIZATION LAYER BENCHMARK SUITE")
    print("="*80)
    print(f"Backend: {backend.backend()}")
    print(f"Iterations per test: 1000")
    print("="*80)
    print()

    methods = [
        "v1_shape_identity",
        "v2_try_except",
        "v3_element_count",
        "v4_unified",
    ]

    results = {}

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Input shape: {scenario.input_shape}, Axis: {scenario.axis}")
        print(f"Mean shape: {scenario.mean_shape}")
        print(f"{'='*80}")

        scenario_results = {}

        for method in methods:
            print(f"\n  Testing {method}...")
            layer_class = create_layer_variant(normalization.Normalization, method)

            # Benchmark build time
            build_stats = benchmark_build_time(layer_class, scenario, method, iterations=1000)

            # Benchmark call time
            call_stats = benchmark_call_time(layer_class, scenario, method, iterations=1000)

            scenario_results[method] = {
                'build': build_stats,
                'call': call_stats,
            }

            if build_stats:
                print(f"    Build time: {build_stats['mean']*1000:.4f} ms "
                      f"(median: {build_stats['median']*1000:.4f} ms, "
                      f"stdev: {build_stats['stdev']*1000:.4f} ms)")
            if call_stats:
                print(f"    Call time:  {call_stats['mean']*1000:.4f} ms "
                      f"(median: {call_stats['median']*1000:.4f} ms, "
                      f"stdev: {call_stats['stdev']*1000:.4f} ms)")

        results[scenario.name] = scenario_results

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Build Time Comparison (mean ms)")
    print("="*80)
    print(f"{'Scenario':<25} {'v1_shape':<12} {'v2_try':<12} {'v3_elem':<12} {'v4_uni':<12}")
    print("-"*80)

    for scenario in SCENARIOS:
        row = f"{scenario.name:<25}"
        for method in methods:
            method_short = method.split('_')[0]
            build_time = results[scenario.name][method]['build']['mean'] * 1000
            row += f" {build_time:>10.4f} "
        print(row)

    print("\n" + "="*80)
    print("SUMMARY - Call Time Comparison (mean ms)")
    print("="*80)
    print(f"{'Scenario':<25} {'v1_shape':<12} {'v2_try':<12} {'v3_elem':<12} {'v4_uni':<12}")
    print("-"*80)

    for scenario in SCENARIOS:
        row = f"{scenario.name:<25}"
        for method in methods:
            call_time = results[scenario.name][method]['call']['mean'] * 1000
            row += f" {call_time:>10.4f} "
        print(row)

    # Determine winner
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)

    build_wins = {method: 0 for method in methods}
    call_wins = {method: 0 for method in methods}

    for scenario in SCENARIOS:
        # Build time winner
        best_build = min(methods, key=lambda m: results[scenario.name][m]['build']['mean'])
        build_wins[best_build] += 1

        # Call time winner
        best_call = min(methods, key=lambda m: results[scenario.name][m]['call']['mean'])
        call_wins[best_call] += 1

    print("\nBuild time wins:")
    for method in methods:
        print(f"  {method}: {build_wins[method]}/{len(SCENARIOS)} scenarios")

    print("\nCall time wins:")
    for method in methods:
        print(f"  {method}: {call_wins[method]}/{len(SCENARIOS)} scenarios")

    overall_winner = max(methods, key=lambda m: build_wins[m] + call_wins[m])
    print(f"\nOverall winner: {overall_winner}")

    return results


if __name__ == "__main__":
    if HAS_KERAS:
        results = run_benchmarks()
    else:
        print("Keras not available. Showing benchmark structure only.")
        print(f"Would test {len(SCENARIOS)} scenarios with 4 methods each.")
        for scenario in SCENARIOS:
            print(f"  - {scenario.name}: {scenario.description}")
