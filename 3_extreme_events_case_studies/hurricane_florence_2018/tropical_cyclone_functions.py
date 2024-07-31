import numpy as np
import xarray as xr
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

def calculate_trajectory(data, latitudes, longitudes, threshold_distance=10, threshold_data_value=100500, neighborhood_size=200):
    """
    Calculate the trajectory of a minimum value across a series of data timesteps.

    This function identifies local minima in a dataset at each timestep, and then tracks the trajectory
    of these minima over time, subject to distance and data value thresholds. The trajectory is defined
    as the series of geographical coordinates (latitude, longitude) where the minima occur.

    Parameters:
    - data (numpy.ndarray): A 3D array of data values, where each 2D slice along the first axis
      represents data for a specific timestep. The array dimensions should be [time, latitude, longitude].
    - latitudes (numpy.ndarray): A 1D array of latitude values corresponding to the latitude dimension
      of the data array.
    - longitudes (numpy.ndarray): A 1D array of longitude values corresponding to the longitude dimension
      of the data array.
    - threshold_distance (float, optional): The maximum allowable distance between consecutive minima in
      the trajectory. If the distance between consecutive minima exceeds this threshold, the trajectory calculation
      is stopped. Defaults to 10.
    - threshold_data_value (float, optional): The maximum allowable data value for a minimum to be included
      in the trajectory. If a minimum exceeds this data value, the trajectory calculation is stopped. Defaults to 100500.
    - neighborhood_size (int, optional): The size of the neighborhood used in the minimum filtering process
      to identify local minima. Defaults to 200.

    Returns:
    - selected_latitudes (list of float): A list of latitude values where the identified minima occur,
      forming the trajectory.
    - selected_longitudes (list of float): A list of longitude values where the identified minima occur,
      forming the trajectory.
    - minima (list of float): A list of data values corresponding to the identified minima at each timestep.
    """
    # Create an empty list to store minima indices and values for each timestep
    minima_indices_per_timestep = []
    minima_value_per_timestep = []

    # Iterate over each timestep
    for timestep_data in data:
        # Check for missing values in the data
        if np.any(np.isnan(timestep_data)):
            continue  # Skip this timestep if it contains missing values

        # Apply minimum filtering to find local minima for the current timestep
        local_minima = minimum_filter(timestep_data, size=neighborhood_size)

        # Identify the smallest local minimum
        local_minima = np.min(local_minima)

        # Find indices where the original data is equal to the filtered result
        minima_indices = np.where(timestep_data == local_minima)

        # Append minima indices and value for the current timestep to the lists
        minima_indices_per_timestep.append(minima_indices)
        minima_value_per_timestep.append(local_minima)

    # Initialize lists to store the selected minima coordinates and values
    selected_latitudes = []
    selected_longitudes = []
    minima = []

    # Iterate over each timestep
    for i, indices in enumerate(minima_indices_per_timestep):
        # Convert indices to coordinates
        minima_coordinates = list(zip(latitudes[indices[0]], longitudes[indices[1]]))

        # Calculate distances from the previous minimum
        if selected_latitudes and selected_longitudes:  # Check if there's a previous minimum
            prev_minimum = (selected_latitudes[-1], selected_longitudes[-1])
            distances = cdist([prev_minimum], minima_coordinates).flatten()
        else:
            distances = np.full(len(minima_coordinates), np.inf)  # Set distances to infinity if no previous minimum

        # Find the index of the minimum distance
        min_distance_index = np.argmin(distances)

        # Check if the minimum distance exceeds the threshold or if the data value exceeds the threshold
        if i > 0 and (np.min(distances) > threshold_distance or minima_value_per_timestep[min_distance_index] > threshold_data_value):
            print(f'break: distance {np.min(distances)}, threshold: {minima_value_per_timestep[min_distance_index]}')
            break
        else:
            # Select the closest minimum
            selected_latitudes.append(minima_coordinates[min_distance_index][0])
            selected_longitudes.append(minima_coordinates[min_distance_index][1])
            minima.append(minima_value_per_timestep[i])

    return selected_latitudes, selected_longitudes, minima

def haversine(dlat, dlon, lat1, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    dlat, dlon, lat1, lat2 = map(np.radians, [dlat, dlon, lat1, lat2])

    # Haversine formula 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def compute_storm_data(msl, tcwv, u100, v100, lat_track, lon_track, lats, lons, rel_lat, rel_lon, res_deg=0.25):
    """
    Compute storm-related data, including the radius, mean sea level pressure, total column water vapor, and wind speed.

    This function calculates the radius from the storm center, mean sea level pressure (MSL), total column water vapor (TCWV), 
    and wind speed (WSP) around a specified storm center based on the provided latitude and longitude tracks.

    Parameters:
    - msl (xarray.DataArray): Mean sea level pressure data.
    - tcwv (xarray.DataArray): Total column water vapor data.
    - u100 (xarray.DataArray): Eastward wind speed at 100 meters above ground.
    - v100 (xarray.DataArray): Northward wind speed at 100 meters above ground.
    - lat_track (float): Latitude of the storm center.
    - lon_track (float): Longitude of the storm center.
    - lats (xarray.DataArray): Latitude grid data.
    - lons (xarray.DataArray): Longitude grid data.
    - rel_lat (xarray.DataArray): Relative latitude positions from the storm center.
    - rel_lon (xarray.DataArray): Relative longitude positions from the storm center.
    - res_deg (float, optional): Grid resolution in degrees. Defaults to 0.25.

    Returns:
    - radius (xarray.DataArray): The radius from the storm center at each relative position.
    - msl_storm (xarray.DataArray): The mean sea level pressure at each relative position.
    - tcwv_storm (xarray.DataArray): The total column water vapor at each relative position.
    - wsp_storm (xarray.DataArray): The wind speed at each relative position.
    """
    # Find the indices of the storm center in the latitude and longitude arrays
    lat_idx = np.where(lats == lat_track)[0][0]
    lon_idx = np.where(lons == lon_track)[0][0]

    # Earth radius in kilometers and resolution in radians
    Re = 6370  # km
    res = res_deg * np.pi / 180.

    # Compute the increment indices for relative positions
    incr_lat = (rel_lat / res_deg).astype('int')
    incr_lon = (rel_lon / res_deg).astype('int')

    # Initialize data arrays for storing computed values
    radius = xr.DataArray(dims=('rel_lat', 'rel_lon'), coords={'rel_lat': rel_lat, 'rel_lon': rel_lon})
    msl_storm = xr.DataArray(dims=('rel_lat', 'rel_lon'), coords={'rel_lat': rel_lat, 'rel_lon': rel_lon})
    tcwv_storm = xr.DataArray(dims=('rel_lat', 'rel_lon'), coords={'rel_lat': rel_lat, 'rel_lon': rel_lon})
    wsp_storm = xr.DataArray(dims=('rel_lat', 'rel_lon'), coords={'rel_lat': rel_lat, 'rel_lon': rel_lon})

    # Iterate over relative latitude and longitude indices to compute values
    for j, i_lat in enumerate(incr_lat):
        for i, i_lon in enumerate(incr_lon):
            lat_i = lat_idx - i_lat
            lon_i = lon_idx + i_lon
            radius[j, i] = np.sqrt((i_lon * res * Re) ** 2 + (i_lat * res * np.cos(lats[lat_i] * np.pi / 180.) * Re) ** 2)
            msl_storm[j, i] = msl[lat_i, lon_i]
            tcwv_storm[j, i] = tcwv[lat_i, lon_i]
            wsp_storm[j, i] = np.sqrt(u100[lat_i, lon_i] ** 2 + v100[lat_i, lon_i] ** 2)
 
    return radius, msl_storm, tcwv_storm, wsp_storm


def transform_to_radius(msl_storm, wsp_storm, radius, incr_radius=np.arange(0,800,25), dr=5):
    """
    This function transforms the relative latitude and longitude to radius and dimension.

    Parameters:
    msl_storm (xarray.DataArray): The mean sea level pressure of the storm.
    wsp_storm (xarray.DataArray): The wind speed of the storm.
    radius (xarray.DataArray): The radius of the storm.
    incr_radius (numpy.ndarray, optional): Incremental radius. Defaults to np.arange(0,800,25).

    Returns:
    msl_storm_radius (xarray.DataArray): The mean sea level pressure of the storm transformed to radius.
    wsp_storm_radius (xarray.DataArray): The wind speed of the storm transformed to radius.
    """

    #this should have the same dims as the original variables but collapsing the last rel_lat and rel_lon
    dims = list(msl_storm.dims[:-2]) + ['radius']
    coords = {dim: msl_storm.coords[dim] for dim in msl_storm.dims[:-2]}
    coords.update({'radius' : incr_radius})
    msl_storm_radius = xr.DataArray(dims=dims, coords=coords)
    wsp_storm_radius = xr.DataArray(dims=dims, coords=coords)
    max_wsp_storm_radius = xr.DataArray(dims=dims, coords=coords)
    
    for i,r in enumerate(incr_radius):
        if i>0:
            #bool = ((radius <= r) & (radius > incr_radius[i-1]))
            bool = ((radius <= r + dr) & (radius > r -dr))
        else:
            bool = (radius <= r)

        msl_nans = (msl_storm * bool)
        msl_nans = msl_nans.where(msl_nans != 0, np.nan)
        msl_storm_radius.loc[{'radius' : r}] = msl_nans.mean(['rel_lat','rel_lon'])

        wsp_nans = (wsp_storm * bool)
        wsp_nans = wsp_nans.where(wsp_nans != 0, np.nan)
        wsp_storm_radius.loc[{'radius' : r}] = wsp_nans.mean(['rel_lat','rel_lon'])
        max_wsp_storm_radius.loc[{'radius' : r}] = wsp_nans.max(['rel_lat','rel_lon'])

    return msl_storm_radius, wsp_storm_radius, max_wsp_storm_radius


def calculate_storm_size_int(windspeeds, radius, target_windspeed):
    """
    Calculate the radius at which the wind speed reaches a target value in a storm.

    This function determines the size of a storm by identifying the radius where the wind speed 
    equals a specified target windspeed. It uses linear interpolation for estimating this radius 
    based on provided windspeed and radius data.

    Parameters:
    - windspeeds (numpy.ndarray): An array of wind speed values, typically ordered from the center
      of the storm outward.
    - radius (numpy.ndarray): An array of radius values corresponding to the distances from the 
      storm's center, matching the wind speeds in position.
    - target_windspeed (float): The wind speed value for which the corresponding radius is sought.

    Returns:
    - float: The interpolated radius where the wind speed equals the target value. If wind speeds are 
      not available or interpolation is not possible, it returns NaN.
    """
    # Check if there are any NaN values in the wind speeds
    if np.isnan(windspeeds).any():
        return np.nan  # Return NaN if any wind speeds are NaN
        
    # Find the index of the maximum wind speed
    max_idx = windspeeds.argmax().item()
        
    # Consider only the radii and wind speeds beyond the maximum wind speed index
    valid_radii = radius[max_idx + 1:]
    valid_windspeeds = windspeeds[max_idx + 1:]
    
    # If no valid wind speeds remain, return NaN
    if valid_windspeeds.size == 0:
        return np.nan

    # Find the index of the closest wind speed to the target wind speed
    closest_idx = np.abs(valid_windspeeds - target_windspeed).argmin()

    # Interpolate to find the radius corresponding to the target wind speed
    try:
        interpolator = interp1d(
            valid_windspeeds[closest_idx-3:closest_idx+5], 
            valid_radii[closest_idx-3:closest_idx+5], 
            kind='linear', 
            fill_value="extrapolate"
        )
        interpolated_radius = interpolator(target_windspeed)
        return interpolated_radius
    except ValueError as e:
        # Handle interpolation errors
        print(f"Interpolation error: {e}")
        return np.nan
