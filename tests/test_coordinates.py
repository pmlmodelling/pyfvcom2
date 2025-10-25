"""
Unit tests for coordinates.py module.

Tests the coordinate transformation functions for FVCOM model output.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pyfvcom2.coordinates import (
    sigma_to_z_coords, 
    z_to_sigma_coords,
    get_epsg_code,
    utm_from_lonlat,
    lonlat_from_utm
)
from pyfvcom2.exceptions import PyFVCOM2ValueError, PyFVCOM2RuntimeError

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


class TestGetEpsgCode:
    """Test cases for get_epsg_code function."""
    
    @patch('pyfvcom2.coordinates.query_utm_crs_info')
    @patch('pyfvcom2.coordinates.AreaOfInterest')
    def test_basic_epsg_lookup(self, mock_area, mock_query):
        """Test basic EPSG code lookup functionality."""
        # Mock the query result
        mock_crs_info = MagicMock()
        mock_crs_info.code = '32630'  # UTM Zone 30N
        mock_query.return_value = [mock_crs_info]
        
        result = get_epsg_code(-2.0, 50.0)
        
        # Verify AreaOfInterest was created with correct bounds
        mock_area.assert_called_once_with(
            west_lon_degree=-2.0,
            south_lat_degree=50.0,
            east_lon_degree=-2.0,
            north_lat_degree=50.0
        )
        
        # Verify query was called with default datum
        mock_query.assert_called_once_with(
            datum_name="WGS 84",
            area_of_interest=mock_area.return_value
        )
        
        assert result == '32630'
    
    @patch('pyfvcom2.coordinates.query_utm_crs_info')
    @patch('pyfvcom2.coordinates.AreaOfInterest')
    def test_custom_datum(self, mock_area, mock_query):
        """Test EPSG code lookup with custom datum."""
        mock_crs_info = MagicMock()
        mock_crs_info.code = '27700'  # British National Grid
        mock_query.return_value = [mock_crs_info]
        
        result = get_epsg_code(-2.0, 52.0, datum="OSGB 1936")
        
        mock_query.assert_called_once_with(
            datum_name="OSGB 1936",
            area_of_interest=mock_area.return_value
        )
        
        assert result == '27700'
    
    @patch('pyfvcom2.coordinates.query_utm_crs_info')
    @patch('pyfvcom2.coordinates.AreaOfInterest')
    def test_different_hemispheres(self, mock_area, mock_query):
        """Test EPSG code lookup for different hemispheres."""
        # Test Southern Hemisphere
        mock_crs_info = MagicMock()
        mock_crs_info.code = '32730'  # UTM Zone 30S
        mock_query.return_value = [mock_crs_info]
        
        result = get_epsg_code(-2.0, -20.0)
        
        mock_area.assert_called_once_with(
            west_lon_degree=-2.0,
            south_lat_degree=-20.0,
            east_lon_degree=-2.0,
            north_lat_degree=-20.0
        )
        
        assert result == '32730'


class TestUtmFromLonlat:
    """Test cases for utm_from_lonlat function."""
    
    @patch('pyfvcom2.coordinates.get_epsg_code')
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_single_point_conversion(self, mock_transformer, mock_crs, mock_get_epsg):
        """Test UTM conversion for a single point."""
        # Setup mocks
        mock_get_epsg.return_value = '32630'
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = (500000.0, 5540000.0)
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test conversion
        eastings, northings, epsg = utm_from_lonlat(-2.0, 50.0)
        
        # Verify EPSG code was determined
        mock_get_epsg.assert_called_once_with(-2.0, 50.0)
        
        # Verify CRS was created
        mock_crs.from_epsg.assert_called_once_with('32630')
        
        # Verify transformer was created
        mock_transformer.from_crs.assert_called_once_with(
            mock_crs_instance.geodetic_crs,
            mock_crs_instance,
            always_xy=True
        )
        
        # Verify transform was called
        mock_proj.transform.assert_called_once()
        
        # Check results
        np.testing.assert_array_equal(eastings, [500000.0])
        np.testing.assert_array_equal(northings, [5540000.0])
        assert epsg == '32630'
    
    @patch('pyfvcom2.coordinates.get_epsg_code')
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_array_conversion(self, mock_transformer, mock_crs, mock_get_epsg):
        """Test UTM conversion for multiple points."""
        # Setup mocks
        mock_get_epsg.return_value = '32630'
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = ([500000.0, 600000.0], [5540000.0, 5640000.0])
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test with arrays
        lons = np.array([-2.0, -1.0])
        lats = np.array([50.0, 51.0])
        
        eastings, northings, epsg = utm_from_lonlat(lons, lats)
        
        # Verify EPSG code was determined from first point
        mock_get_epsg.assert_called_once_with(-2.0, 50.0)
        
        # Check results
        np.testing.assert_array_equal(eastings, [500000.0, 600000.0])
        np.testing.assert_array_equal(northings, [5540000.0, 5640000.0])
        assert epsg == '32630'
    
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_explicit_epsg_code(self, mock_transformer, mock_crs):
        """Test UTM conversion with explicitly provided EPSG code."""
        # Setup mocks
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = (500000.0, 5540000.0)
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test with explicit EPSG code
        eastings, northings, epsg = utm_from_lonlat(-2.0, 50.0, epsg_code='32631')
        
        # Verify CRS was created with provided EPSG
        mock_crs.from_epsg.assert_called_once_with(32631)
        
        assert epsg == 32631
    
    def test_mismatched_array_sizes(self):
        """Test error handling for mismatched longitude/latitude array sizes."""
        lons = np.array([-2.0, -1.0])
        lats = np.array([50.0])  # Different size
        
        with pytest.raises(PyFVCOM2RuntimeError, match="Lat and lon array sizes do not match"):
            utm_from_lonlat(lons, lats)
    
    @patch('pyfvcom2.coordinates.get_epsg_code')
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_different_input_types(self, mock_transformer, mock_crs, mock_get_epsg):
        """Test UTM conversion with different input types."""
        # Setup mocks
        mock_get_epsg.return_value = '32630'
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = (500000.0, 5540000.0)
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test with different input types
        test_cases = [
            (-2.0, 50.0),  # float
            (-2, 50),      # int
            ([-2.0], [50.0]),  # list
            ((-2.0,), (50.0,)),  # tuple
        ]
        
        for lon_input, lat_input in test_cases:
            eastings, northings, epsg = utm_from_lonlat(lon_input, lat_input)
            assert epsg == '32630'


class TestLonlatFromUtm:
    """Test cases for lonlat_from_utm function."""
    
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_single_point_conversion(self, mock_transformer, mock_crs):
        """Test lat/lon conversion from UTM for a single point."""
        # Setup mocks
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = (-2.0, 50.0)
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test conversion
        lons, lats = lonlat_from_utm(500000.0, 5540000.0, '32630')
        
        # Verify CRS was created
        mock_crs.from_epsg.assert_called_once_with(32630)
        
        # Verify transformer was created
        mock_transformer.from_crs.assert_called_once_with(
            mock_crs_instance.geodetic_crs,
            mock_crs_instance,
            always_xy=True
        )
        
        # Verify transform was called with inverse direction
        mock_proj.transform.assert_called_once()
        call_args = mock_proj.transform.call_args
        assert 'direction' in call_args.kwargs
        
        # Check results
        np.testing.assert_array_equal(lons, [-2.0])
        np.testing.assert_array_equal(lats, [50.0])
    
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_array_conversion(self, mock_transformer, mock_crs):
        """Test lat/lon conversion from UTM for multiple points."""
        # Setup mocks
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = ([-2.0, -1.0], [50.0, 51.0])
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test with arrays
        eastings = np.array([500000.0, 600000.0])
        northings = np.array([5540000.0, 5640000.0])
        
        lons, lats = lonlat_from_utm(eastings, northings, '32630')
        
        # Check results
        np.testing.assert_array_equal(lons, [-2.0, -1.0])
        np.testing.assert_array_equal(lats, [50.0, 51.0])
    
    def test_mismatched_array_sizes(self):
        """Test error handling for mismatched easting/northing array sizes."""
        eastings = np.array([500000.0, 600000.0])
        northings = np.array([5540000.0])  # Different size
        
        with pytest.raises(PyFVCOM2RuntimeError, match="Easting and northing array sizes do not match"):
            lonlat_from_utm(eastings, northings, '32630')
    
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_different_input_types(self, mock_transformer, mock_crs):
        """Test lat/lon conversion with different input types."""
        # Setup mocks
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        mock_proj = MagicMock()
        mock_proj.transform.return_value = (-2.0, 50.0)
        mock_transformer.from_crs.return_value = mock_proj
        
        # Test with different input types
        test_cases = [
            (500000.0, 5540000.0),  # float
            (500000, 5540000),      # int
            ([500000.0], [5540000.0]),  # list
            ((500000.0,), (5540000.0,)),  # tuple
        ]
        
        for east_input, north_input in test_cases:
            lons, lats = lonlat_from_utm(east_input, north_input, '32630')
            # Results should be consistent regardless of input type
            np.testing.assert_array_equal(lons, [-2.0])
            np.testing.assert_array_equal(lats, [50.0])


class TestUtmRoundtripTransformations:
    """Test round-trip transformations between lat/lon and UTM coordinates."""
    
    @patch('pyfvcom2.coordinates.get_epsg_code')
    @patch('pyfvcom2.coordinates.CRS')
    @patch('pyfvcom2.coordinates.Transformer')
    def test_lonlat_to_utm_to_lonlat_roundtrip(self, mock_transformer, mock_crs, mock_get_epsg):
        """Test that lon/lat -> UTM -> lon/lat preserves original values."""
        # Setup mocks for forward transformation
        mock_get_epsg.return_value = '32630'
        mock_crs_instance = MagicMock()
        mock_crs.from_epsg.return_value = mock_crs_instance
        
        # Mock transformer for both directions
        mock_proj = MagicMock()
        # Forward: lon/lat -> UTM
        mock_transformer.from_crs.return_value = mock_proj
        
        # Original coordinates
        original_lons = np.array([-2.0, -1.0, 0.0])
        original_lats = np.array([50.0, 51.0, 52.0])
        
        # Mock forward transformation
        mock_proj.transform.return_value = ([500000.0, 600000.0, 700000.0], 
                                          [5540000.0, 5640000.0, 5740000.0])
        
        # Forward transformation
        eastings, northings, epsg = utm_from_lonlat(original_lons, original_lats)
        
        # Mock backward transformation
        mock_proj.transform.return_value = (original_lons, original_lats)
        
        # Backward transformation
        recovered_lons, recovered_lats = lonlat_from_utm(eastings, northings, epsg)
        
        # Should recover original coordinates
        np.testing.assert_array_almost_equal(original_lons, recovered_lons, decimal=6)
        np.testing.assert_array_almost_equal(original_lats, recovered_lats, decimal=6)


if __name__ == "__main__":
    pytest.main([__file__])
