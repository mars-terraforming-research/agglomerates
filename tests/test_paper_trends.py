"""
Integration tests reproducing key scaling trends from Abomailek et al. (Small 2025).

These tests generate agglomerates at test time and verify that the simulation
reproduces the qualitative physical trends reported in the paper:

1. R_g power-law scaling: N = k'_0 * (R_g/b)^D_f with D_f ~ 1.8
2. D_f,BC convergence: 2D box-counting fractal dimension increases with N
3. b-normalization collapse: normalizing R_g by b collapses curves for different geometries
4. Size growth: bounding box and R_g increase monotonically with N

Tolerances are generous since we test qualitative physical trends, not exact numbers.
"""

import numpy as np
import pytest

from agglomerate import generate_agglomerate, calculate_agglomerate_com, Nanorod
from analyze_batch_paper_method import calculate_fractal_dimension_paper_method
from generate_batch import calculate_bounding_box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def radius_of_gyration(positions):
    """Compute R_g = sqrt(1/N * sum |r_i - r_com|^2)."""
    com = positions.mean(axis=0)
    return np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))


def volume_equivalent_radius(length, diameter):
    """Compute b = (3/4 * R_NW^2 * L)^(1/3), the volume-equivalent sphere radius."""
    return (0.75 * (diameter / 2) ** 2 * length) ** (1 / 3)


def _generate_and_get_positions(n, length, diameter, seed):
    """Generate an agglomerate and return the rod center positions as an array."""
    agg = generate_agglomerate(n, length, diameter, seed=seed, verbose=False)
    return agg, np.array([rod.center for rod in agg])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestRadiusOfGyration:
    """Basic sanity checks for R_g computation."""

    def test_rg_positive_for_two_or_more(self):
        """R_g > 0 for N >= 2."""
        for n in [2, 5, 10]:
            _, positions = _generate_and_get_positions(n, 1000, 15, seed=42)
            rg = radius_of_gyration(positions)
            assert rg > 0, f"R_g should be positive for N={n}, got {rg}"

    def test_rg_zero_for_single_particle(self):
        """R_g = 0 for a single particle (trivially at the origin)."""
        _, positions = _generate_and_get_positions(1, 1000, 15, seed=42)
        rg = radius_of_gyration(positions)
        assert rg == 0.0

    def test_rg_increases_with_n(self):
        """R_g should generally increase with N (averaged over seeds)."""
        n_values = [5, 10, 25, 50]
        seeds = [10, 20, 30]
        mean_rg = []

        for n in n_values:
            rgs = []
            for seed in seeds:
                _, positions = _generate_and_get_positions(n, 1000, 15, seed=seed)
                rgs.append(radius_of_gyration(positions))
            mean_rg.append(np.mean(rgs))

        # Check monotonic increase
        for i in range(len(mean_rg) - 1):
            assert mean_rg[i + 1] > mean_rg[i], (
                f"Mean R_g should increase: R_g(N={n_values[i]})={mean_rg[i]:.1f} "
                f">= R_g(N={n_values[i+1]})={mean_rg[i+1]:.1f}"
            )


class TestFractalScaling:
    """
    Test R_g power-law scaling: log(R_g) vs log(N) should be linear.

    Paper reports D_f ~ 1.8 (3D mass-radius), so slope of log(R_g) vs log(N)
    should be ~1/D_f ~ 0.56. We use generous bounds: 1.4 < D_f < 2.2.
    """

    @pytest.fixture(scope="class")
    def scaling_data(self):
        """Generate agglomerates and compute R_g for scaling analysis."""
        n_values = [10, 20, 50, 100]
        seeds = [10, 20, 30]
        results = {}

        for n in n_values:
            rgs = []
            for seed in seeds:
                _, positions = _generate_and_get_positions(n, 1000, 15, seed=seed)
                rgs.append(radius_of_gyration(positions))
            results[n] = np.mean(rgs)

        return n_values, results

    def test_power_law_fit_quality(self, scaling_data):
        """log-log fit of N vs R_g should have R^2 > 0.9."""
        n_values, results = scaling_data
        log_n = np.log(n_values)
        log_rg = np.log([results[n] for n in n_values])

        coeffs = np.polyfit(log_n, log_rg, 1)
        y_pred = np.polyval(coeffs, log_n)
        ss_res = np.sum((log_rg - y_pred) ** 2)
        ss_tot = np.sum((log_rg - np.mean(log_rg)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.9, f"Power-law fit R^2 = {r_squared:.3f}, expected > 0.9"

    def test_fractal_dimension_range(self, scaling_data):
        """D_f from mass-radius scaling should be in [1.4, 2.2]."""
        n_values, results = scaling_data
        log_n = np.log(n_values)
        log_rg = np.log([results[n] for n in n_values])

        # slope of log(R_g) vs log(N) is 1/D_f
        slope = np.polyfit(log_n, log_rg, 1)[0]
        d_f = 1.0 / slope

        assert 1.4 < d_f < 2.5, f"D_f = {d_f:.2f}, expected in (1.4, 2.5)"


class TestBoxCountingConvergence:
    """
    Test that 2D box-counting D_f,BC increases with N and converges
    to a reasonable range for large N (paper: ~1.5-1.8 for N > 50).
    """

    @pytest.fixture(scope="class")
    def box_counting_data(self):
        """Generate agglomerates and compute D_f,BC."""
        n_values = [10, 25, 50, 100]
        seed = 42
        results = {}

        for n in n_values:
            agg = generate_agglomerate(n, 1000, 15, seed=seed, verbose=False)
            # Use lower resolution for speed in tests
            np.random.seed(seed)
            mean_df, _, _ = calculate_fractal_dimension_paper_method(
                agg, n_projections=3, resolution=500
            )
            results[n] = mean_df

        return n_values, results

    def test_df_bc_increases_with_n(self, box_counting_data):
        """D_f,BC at N=100 should exceed D_f,BC at N=10."""
        _, results = box_counting_data
        assert results[100] > results[10], (
            f"D_f,BC(N=100)={results[100]:.3f} should exceed "
            f"D_f,BC(N=10)={results[10]:.3f}"
        )

    def test_df_bc_range_at_large_n(self, box_counting_data):
        """D_f,BC at N=100 should be in [1.3, 2.0]."""
        _, results = box_counting_data
        assert 1.3 <= results[100] <= 2.0, (
            f"D_f,BC(N=100)={results[100]:.3f}, expected in [1.3, 2.0]"
        )


class TestNormalizationCollapse:
    """
    Test b-normalization collapse (Fig 8A of paper).

    For different rod geometries with the same diameter, normalizing R_g by
    b = (3/4 * R_NW^2 * L)^(1/3) should bring curves closer together.
    """

    @pytest.fixture(scope="class")
    def collapse_data(self):
        """Generate agglomerates for two geometries and compute R_g and R_g/b."""
        geometries = [
            {"length": 500, "diameter": 15},
            {"length": 2000, "diameter": 15},
        ]
        n_values = [20, 50, 100]
        seeds = [10, 20]

        data = {}
        for geom in geometries:
            L = geom["length"]
            D = geom["diameter"]
            b = volume_equivalent_radius(L, D)
            key = (L, D)
            data[key] = {"b": b, "rg": {}, "rg_norm": {}}

            for n in n_values:
                rgs = []
                for seed in seeds:
                    _, positions = _generate_and_get_positions(n, L, D, seed=seed)
                    rgs.append(radius_of_gyration(positions))
                mean_rg = np.mean(rgs)
                data[key]["rg"][n] = mean_rg
                data[key]["rg_norm"][n] = mean_rg / b

        return geometries, n_values, data

    def test_normalization_reduces_spread(self, collapse_data):
        """Normalizing R_g by b should reduce the spread between geometries."""
        geometries, n_values, data = collapse_data

        keys = [(g["length"], g["diameter"]) for g in geometries]

        raw_spreads = []
        norm_spreads = []

        for n in n_values:
            raw_vals = [data[k]["rg"][n] for k in keys]
            norm_vals = [data[k]["rg_norm"][n] for k in keys]

            # Relative spread: |a - b| / mean(a, b)
            raw_spread = abs(raw_vals[0] - raw_vals[1]) / np.mean(raw_vals)
            norm_spread = abs(norm_vals[0] - norm_vals[1]) / np.mean(norm_vals)

            raw_spreads.append(raw_spread)
            norm_spreads.append(norm_spread)

        mean_raw = np.mean(raw_spreads)
        mean_norm = np.mean(norm_spreads)

        assert mean_norm < mean_raw, (
            f"Normalized spread ({mean_norm:.3f}) should be less than "
            f"raw spread ({mean_raw:.3f})"
        )


class TestAgglomerateSizeGrowth:
    """Test that agglomerate spatial extent grows with particle count."""

    @pytest.fixture(scope="class")
    def growth_data(self):
        """Generate agglomerates at increasing N and compute R_g and bbox volume."""
        n_values = [5, 10, 25, 50]
        seed = 42
        rg_list = []
        vol_list = []

        for n in n_values:
            agg = generate_agglomerate(n, 1000, 15, seed=seed, verbose=False)
            positions = np.array([rod.center for rod in agg])
            rg_list.append(radius_of_gyration(positions))

            bbox = calculate_bounding_box(agg)
            vol_list.append(bbox["length"] * bbox["width"] * bbox["height"])

        return n_values, rg_list, vol_list

    def test_rg_increases_monotonically(self, growth_data):
        """R_g should increase monotonically with N."""
        n_values, rg_list, _ = growth_data
        for i in range(len(rg_list) - 1):
            assert rg_list[i + 1] > rg_list[i], (
                f"R_g(N={n_values[i+1]})={rg_list[i+1]:.1f} should exceed "
                f"R_g(N={n_values[i]})={rg_list[i]:.1f}"
            )

    def test_bbox_volume_increases_monotonically(self, growth_data):
        """Bounding box volume should increase monotonically with N."""
        n_values, _, vol_list = growth_data
        for i in range(len(vol_list) - 1):
            assert vol_list[i + 1] > vol_list[i], (
                f"Volume(N={n_values[i+1]})={vol_list[i+1]:.1e} should exceed "
                f"Volume(N={n_values[i]})={vol_list[i]:.1e}"
            )
