import xarray as xr
import numpy as np
import datetime
import pandas as pd

def flip_lon_360_2_180(var_360, lon):
    """
    This function shifts the longitude dimension from [0,360] to [-180,180].
    """
    var_180 = var_360.assign_coords(lon=(lon + 180) % 360 - 180)
    var_180 = var_180.sortby(var_180.lon)

    return var_180

def from_init_time_to_leadtime(var_init_time, init_time_min, lead_time_range, time_range):
    """
    This function creates a xarray DataArray for a given variable and fills it with values based on the provided time range and lead time range.

    Parameters:
    var_init_time (xarray.DataArray): The initial time data for the variable.
    init_time_min (datetime): The minimum initial time.
    lead_time_range (numpy.ndarray): The range of lead times.
    time_range (pandas.DatetimeIndex): The range of times.

    Returns:
    xarray.DataArray: The created and filled DataArray for the variable.
    """

    # Create a new DataArray for the variable with extra dimensions
    dims = ["lead_time", "time"] + list(var_init_time.dims)[2:]
    coords = {"lead_time": lead_time_range, "time": time_range}
    coords.update({dim: var_init_time.coords[dim] for dim in var_init_time.dims[2:]})
    var_leadtime = xr.DataArray(dims=dims, coords=coords)
    
    # Loop to assign values
    for t in time_range:
        for lt in lead_time_range:
            # Convert the numpy datetime64 object to a Python datetime object
            t_datetime = pd.to_datetime(str(t.values))

            # Subtract the timedelta
            it = t_datetime - datetime.timedelta(hours=int(lt))

            # Assign NaN if the initial time is less than the minimum initial time
            if it < init_time_min:
                var_leadtime.loc[{"lead_time": lt, "time": t}] = np.nan
            else:
                try:
                    # Assign the value from the initial time data for the variable
                    var_leadtime.loc[{"lead_time": lt, "time": t}] = var_init_time.sel(init_time=it, time=t)
                except:
                    # Assign NaN if the value cannot be assigned
                    var_leadtime.loc[{"lead_time": lt, "time": t}] = np.nan

    return var_leadtime

import regionmask

def add_country_mask(ds: xr.Dataset, country: str="Spain") -> xr.Dataset:

    # get countries mask
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110

    # create mask variable
    mask = countries.mask_3D(ds)

    # select Spain mask
    var_name = country.lower()
    ds[f"{var_name}_mask"] = mask.isel(region=(mask.names=="Spain")).squeeze().astype(np.int16)

    return ds

def add_land_mask(ds: xr.Dataset) -> xr.Dataset:

    # get land-sea-mask mask
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    # create land mask variable
    ds["land_mask"] = land_110.mask_3D(ds).squeeze().astype(np.int16)

    return ds

def load_data(var, init_times, root, extension='', model='fcnv2'):
    """
    Load and concatenate data from multiple NetCDF files into a single xarray Dataset.

    This function constructs file paths for each initialization time, loads the specified variable from each file,
    and concatenates them along a new dimension called 'init_time'.

    Parameters:
        var (str): The name of the variable to load from the NetCDF files.
        init_times (list of datetime.datetime): List of initialization times used to construct file paths.
        root (str): The root directory where the NetCDF files are located.
        extension (str, optional): An optional type suffix to include in the file path (default is '').
        model (str, optional): The model identifier to include in the file path (default is 'fcnv2').

    Returns:
        xarray.Dataset: An xarray Dataset containing the concatenated data from all files.
            The dataset will have a new coordinate dimension 'init_time' corresponding to the initialization times.

    Example:
        >>> from datetime import datetime
        >>> init_times = [datetime(2020, 1, 1, 0), datetime(2020, 1, 2, 0)]
        >>> root = '/path/to/data'
        >>> data = load_data('temperature', init_times, root, type='daily', model='fcnv2')
        >>> print(data)
    """

    var_inits = []
    for t0 in init_times:
        yyyymmddhh = t0.strftime('%Y%m%d%H')    
        file = f'{root}/{yyyymmddhh}/{var}_{model}_{extension}{yyyymmddhh}.nc'
        ds = xr.open_dataset(file)[var]
        var_inits.append(ds)
    merged_dataset = xr.concat(var_inits, dim='init_time').squeeze()

    return merged_dataset.assign_coords(init_time=init_times)

def area_average(var):
    """
    Compute the area-weighted global average of a variable, ignoring missing values.

    This function calculates the global average of a variable using latitude-dependent weights 
    while excluding missing values (NaNs) from the calculation.

    Parameters:
        var (xarray.DataArray): Input data with 'lat' and 'lon' dimensions, may contain NaNs.

    Returns:
        xarray.DataArray: Area-weighted global average, averaged over longitude, excluding NaNs.
    """
    # Extract latitude values from the DataArray
    lat = var.lat
    
    # Compute area weights based on latitude
    weights = np.cos(lat * np.pi / 180)
    
    # Ensure weights and var are aligned
    weights_da = xr.DataArray(weights, coords=[var.lat], dims=['lat']).broadcast_like(var)
    weights_da = xr.where(np.isnan(var), np.nan, weights_da)
    
    # Calculate the weighted variable and the total weight
    weighted_var = var * weights_da
    weighted_sum = weighted_var.sum('lat', skipna=True)

    total_weight = weights_da.sum('lat', skipna=True)
    
    # Calculate the global mean, ignoring NaNs
    area_average = (weighted_sum / total_weight).mean('lon', skipna=True)
    
    return area_average
