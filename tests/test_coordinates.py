"""
Unit tests for coordinates.py module.

Tests the coordinate transformation functions for FVCOM model output.
"""

import numpy as np
import pytest
from pyfvcom2.coordinates import sigma_to_z_coords, z_to_sigma_coords
from pyfvcom2.exceptions import PyFVCOM2ValueError

class TestSigmaToZCoords:
    """Test cases for sigma_to_z_coords function."""
    
    def test_basic_transformation(self):
        """Test basic sigma to z coordinate transformation."""
        # Simple test case with known values
        sigma_coords = np.array([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0]])  # 3 levels, 2 points
        zeta = np.array([1.0, 2.0])  # Sea surface elevation
        bathymetry = np.array([10.0, 20.0])  # Water depth
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # Expected calculations:
        # Point 0: h=10, zet=1, total_depth=11
        # sigma=-1.0: z = 1 + 11 * (-1.0) = -10
        # sigma=-0.5: z = 1 + 11 * (-0.5) = -4.5  
        # sigma=0.0:  z = 1 + 11 * (0.0) = 1
        expected_point_0 = [-10.0, -4.5, 1.0]
        
        # Point 1: h=20, zet=2, total_depth=22
        # sigma=-1.0: z = 2 + 22 * (-1.0) = -20
        # sigma=-0.5: z = 2 + 22 * (-0.5) = -9
        # sigma=0.0:  z = 2 + 22 * (0.0) = 2
        expected_point_1 = [-20.0, -9.0, 2.0]
        
        np.testing.assert_array_almost_equal(result[:, 0], expected_point_0)
        np.testing.assert_array_almost_equal(result[:, 1], expected_point_1)
    
    def test_zero_zeta(self):
        """Test transformation with zero sea surface elevation."""
        sigma_coords = np.array([[-1.0, -1.0], [0.0, 0.0]])
        zeta = np.array([0.0, 0.0])
        bathymetry = np.array([5.0, 10.0])
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # With zeta=0: z = 0 + (h + 0) * sigma = h * sigma
        expected = np.array([[-5.0, -10.0], [0.0, 0.0]])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_negative_zeta(self):
        """Test transformation with negative sea surface elevation."""
        sigma_coords = np.array([[-1.0], [-0.5], [0.0]])
        zeta = np.array([-1.0])  # Below mean sea level
        bathymetry = np.array([10.0])
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # h=10, zet=-1, total_depth=9
        # sigma=-1.0: z = -1 + 9 * (-1.0) = -10
        # sigma=-0.5: z = -1 + 9 * (-0.5) = -5.5
        # sigma=0.0:  z = -1 + 9 * (0.0) = -1
        expected = np.array([[-10.0], [-5.5], [-1.0]])
        
        np.testing.assert_array_almost_equal(result, expected)
    
        """Test that output shape matches input sigma_coords shape."""
        n_levels, n_points = 5, 3
        sigma_coords = np.random.rand(n_levels, n_points) * -1  # Negative values
        zeta = np.random.rand(n_points)
        bathymetry = np.random.rand(n_points) * 10 + 1  # Positive depths
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        assert result.shape == (n_levels, n_points)
    
    def test_single_point(self):
        """Test transformation for a single horizontal point."""
        sigma_coords = np.array([[-1.0], [-0.5], [0.0]])
        zeta = np.array([0.5])
        bathymetry = np.array([8.0])
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # h=8, zet=0.5, total_depth=8.5
        expected = np.array([[-8.0], [-3.75], [0.5]])
        
        np.testing.assert_array_almost_equal(result, expected)


class TestZToSigmaCoords:
    """Test cases for z_to_sigma_coords function."""
    
    def test_basic_transformation(self):
        """Test basic z to sigma coordinate transformation."""
        z_coords = np.array([[-10.0, -20.0], [-4.5, -9.0], [1.0, 2.0]])
        zeta = np.array([1.0, 2.0])
        bathymetry = np.array([10.0, 20.0])
        
        result = z_to_sigma_coords(z_coords, zeta, bathymetry)
        
        # Point 0: h=10, zet=1, total_depth=11
        # z=-10: sigma = (-10 - 1) / 11 = -1.0
        # z=-4.5: sigma = (-4.5 - 1) / 11 = -0.5
        # z=1: sigma = (1 - 1) / 11 = 0.0
        expected_point_0 = [-1.0, -0.5, 0.0]
        
        # Point 1: h=20, zet=2, total_depth=22
        # z=-20: sigma = (-20 - 2) / 22 = -1.0
        # z=-9: sigma = (-9 - 2) / 22 = -0.5
        # z=2: sigma = (2 - 2) / 22 = 0.0
        expected_point_1 = [-1.0, -0.5, 0.0]
        
        np.testing.assert_array_almost_equal(result[:, 0], expected_point_0)
        np.testing.assert_array_almost_equal(result[:, 1], expected_point_1)
    
    def test_zero_zeta(self):
        """Test transformation with zero sea surface elevation."""
        z_coords = np.array([[-5.0, -10.0], [0.0, 0.0]])
        zeta = np.array([0.0, 0.0])
        bathymetry = np.array([5.0, 10.0])
        
        result = z_to_sigma_coords(z_coords, zeta, bathymetry)
        
        # With zeta=0: sigma = (z - 0) / h = z / h
        expected = np.array([[-1.0, -1.0], [0.0, 0.0]])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_output_shape(self):
        """Test that output shape matches input z_coords shape."""
        n_levels, n_points = 4, 6
        z_coords = np.random.rand(n_levels, n_points) * -5  # Negative depths
        zeta = np.random.rand(n_points) * 2 - 1  # Between -1 and 1
        bathymetry = np.random.rand(n_points) * 10 + 2  # Positive depths
        
        result = z_to_sigma_coords(z_coords, zeta, bathymetry)
        
        assert result.shape == (n_levels, n_points)
    
    def test_division_by_zero_protection(self):
        """Test behavior when total depth approaches zero."""
        z_coords = np.array([[0.0], [0.0]])
        zeta = np.array([0.0])
        bathymetry = np.array([0.0])  # Zero depth case
        
        # This should result in division by zero - the function should handle this
        # or we should test that it raises an appropriate warning/error
        with pytest.warns(RuntimeWarning):
            result = z_to_sigma_coords(z_coords, zeta, bathymetry)
            # Result will contain inf or nan values
            assert np.any(np.isnan(result) | np.isinf(result))


class TestRoundTripTransformations:
    """Test round-trip transformations between sigma and z coordinates."""
    
    def test_sigma_to_z_to_sigma_roundtrip(self):
        """Test that sigma -> z -> sigma preserves original values."""
        # Create test data
        sigma_coords = np.array([
            [-1.0, -1.0, -1.0],
            [-0.6, -0.6, -0.6], 
            [-0.2, -0.2, -0.2],
            [0.0, 0.0, 0.0]
        ])
        zeta = np.array([0.5, -0.3, 1.2])
        bathymetry = np.array([8.0, 15.0, 5.0])
        
        # Forward transformation
        z_coords = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # Backward transformation
        sigma_coords_recovered = z_to_sigma_coords(z_coords, zeta, bathymetry)
        
        # Should recover original sigma coordinates
        np.testing.assert_array_almost_equal(sigma_coords, sigma_coords_recovered, decimal=6)
    
    def test_z_to_sigma_to_z_roundtrip(self):
        """Test that z -> sigma -> z preserves original values."""
        # Create realistic z coordinates (negative depths to positive surface)
        z_coords = np.array([
            [-10.0, -20.0, -5.0],
            [-6.0, -12.0, -3.0],
            [-2.0, -4.0, -1.0],
            [1.0, 2.0, 0.5]
        ])
        zeta = np.array([1.0, 2.0, 0.5])
        bathymetry = np.array([10.0, 20.0, 5.0])
        
        # Forward transformation
        sigma_coords = z_to_sigma_coords(z_coords, zeta, bathymetry)
        
        # Backward transformation  
        z_coords_recovered = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # Should recover original z coordinates
        np.testing.assert_array_almost_equal(z_coords, z_coords_recovered, decimal=6)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_arrays(self):
        """Test with larger arrays to ensure performance is reasonable."""
        n_levels, n_points = 50, 1000
        sigma_coords = np.linspace(-1, 0, n_levels).reshape(-1, 1) * np.ones((1, n_points))
        zeta = np.random.rand(n_points) * 2 - 1
        bathymetry = np.random.rand(n_points) * 50 + 5
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        assert result.shape == (n_levels, n_points)
        assert np.all(np.isfinite(result))
    
    def test_extreme_bathymetry(self):
        """Test with very deep and very shallow water."""
        sigma_coords = np.array([[-1.0, -1.0], [0.0, 0.0]])
        zeta = np.array([0.0, 0.0])
        bathymetry = np.array([0.1, 1000.0])  # Very shallow and very deep
        
        result = sigma_to_z_coords(sigma_coords, zeta, bathymetry)
        
        # Check that results are reasonable
        assert result[0, 0] == pytest.approx(-0.1)  # Bottom of shallow water
        assert result[1, 0] == pytest.approx(0.0)   # Surface of shallow water
        assert result[0, 1] == pytest.approx(-1000.0)  # Bottom of deep water
        assert result[1, 1] == pytest.approx(0.0)   # Surface of deep water
    
    def test_input_validation_shapes(self):
        """Test that functions handle mismatched input shapes appropriately."""
        sigma_coords = np.array([[-1.0], [0.0]])
        zeta = np.array([0.0, 1.0])  # Wrong size
        bathymetry = np.array([10.0])
        
        # This should raise an IndexError or similar when trying to access mismatched indices
        with pytest.raises(PyFVCOM2ValueError):
            sigma_to_z_coords(sigma_coords, zeta, bathymetry)


if __name__ == "__main__":
    pytest.main([__file__])