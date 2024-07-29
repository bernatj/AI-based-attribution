import metpy
from metpy.units import units
import numpy as np
import xarray as xr
import datetime as dt
import os
import logging

# ----------------- CONFIGURATION OPTIONS ----------------- #
# Define time range and parameters
time_i = dt.datetime(2018, 7, 22, 0)  # Start date
time_f = dt.datetime(2018, 7, 22, 18)  # End date
delta_h = 6  # Time step in hours

path_delta_mm = '../DATA/CMIP6-delta-climatiologies/interpolated-2.5deg-multimodel/'  # Path to multimodel delta files
path_ic = '/home/bernatj/Data/ai-forecasts/input/grib/'  # Path to initial condition files
outputdir = '/home/bernatj/Data/ai-forecasts/input/netcdf/'  # Output directory

ai_model = 'fcnv2'  # Model name (e.g., 'pangu, fcnv2')
grib_opt = True  # Whether the initial condition files are in GRIB format (output always in netcdf)
plev_dim_name = 'isobaricInhPa'  # Name of the pressure level dimension 

levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]  # Pressure levels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- FUNCTION DEFINITIONS ----------------- #


def adjust_geopotential(z, t, r, sp=False, plev_dim_name='isobaricInhPa'):
    """
    Adjust geopotential based on temperature and humidity.

    Parameters:
        z (xarray.DataArray): Geopotential data.
        t (xarray.DataArray): Temperature data.
        r (xarray.DataArray): Humidity data (relative or specific).
        sp (bool): If True, 'r' is specific humidity; otherwise, it's relative humidity.
        plev_dim_name (str): Name of the pressure level dimension.

    Returns:
        xarray.DataArray: Adjusted geopotential.
    """
    logger.debug("Adjusting geopotential")
    R_d = 287.05  # Gas constant for dry air [J K-1 kg-1]
    pressure = z[plev_dim_name].broadcast_like(z)
    lnp = np.log(pressure)

    # Calculate mixing ratio
    if not sp:
        w = metpy.calc.mixing_ratio_from_relative_humidity(pressure * units.hPa, t * units.K, r / 100)
    else:
        w = r / (1 - r)

    # Calculate virtual temperature
    tv = metpy.calc.virtual_temperature(t * units.K, w).values

    z_new = z.copy()
    # Adjust geopotential using hydrostatic balance and ideal gas law
    for k in range(len(z[plev_dim_name]) - 1):
        tv_k12 = (tv[k] + tv[k + 1]) / 2
        z_new[k + 1] = z_new[k] - (lnp[k + 1] - lnp[k]) * R_d * tv_k12

    return z_new

def apply_delta_to_initial_condition_cmip(initial_condition_files, ds_cmip_deltas, surf_vars, pl_vars, mode='subtract', factor=1, grib=True, plev_dim_name='isobaricInhPa'):
    """
    Apply pseudo global warming deltas to an initial condition dataset.

    Parameters:
        initial_condition_files (dict): Paths to initial condition files with 'surface' and 'pressure' keys.
        ds_cmip_deltas (xarray.Dataset): Dataset containing delta values.
        surf_vars (list): Surface variables to adjust.
        pl_vars (list): Pressure level variables to adjust.
        mode (str): Mode of delta application ('subtract' or 'add'). Default is 'subtract'.
        factor (float): Factor to multiply the deltas by. Default is 1.
        grib (bool): Whether the input files are in GRIB format. Default is True.
        plev_dim_name (str): Name of the pressure level dimension.

    Returns:
        tuple: Modified surface and pressure level initial condition datasets.
    """
    logger.info("Applying deltas to initial condition")
    
    # Load surface data
    engine = 'cfgrib' if grib else None
    logger.debug(f"Opening surface dataset: {initial_condition_files['surface']}")
    initial_condition_S = xr.open_dataset(initial_condition_files['surface'], engine=engine).squeeze()
    initial_condition_S = initial_condition_S.assign_coords(longitude=initial_condition_S["longitude"] % 360).sortby("longitude")

    # Apply deltas to surface variables
    for var in surf_vars:
        if mode == 'subtract':
            initial_condition_S[var] -= ds_cmip_deltas[var] * factor
        elif mode == 'add':
            initial_condition_S[var] += ds_cmip_deltas[var] * factor

    # Load pressure data
    logger.debug(f"Opening pressure dataset: {initial_condition_files['pressure']}")
    initial_condition_P = xr.open_dataset(initial_condition_files['pressure'], engine=engine).squeeze()
    initial_condition_P = initial_condition_P.sortby(plev_dim_name, ascending=False).transpose(plev_dim_name, 'latitude', 'longitude')
    initial_condition_P = initial_condition_P.assign_coords(longitude=initial_condition_P["longitude"] % 360).sortby("longitude")

    # Adjust geopotential height before applying deltas
    if 'q' in pl_vars:
        logger.debug("Adjusting geopotential height with specific humidity")
        z_baro_before = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['q'], sp=True, plev_dim_name=plev_dim_name)
    elif 'r' in pl_vars:
        logger.debug("Adjusting geopotential height with relative humidity")
        z_baro_before = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['r'], plev_dim_name=plev_dim_name)

    # Apply deltas to pressure level variables
    for var in pl_vars:
        if mode == 'subtract':
            initial_condition_P[var] -= ds_cmip_deltas[var].rename({'level': plev_dim_name}) * factor
        elif mode == 'add':
            initial_condition_P[var] += ds_cmip_deltas[var].rename({'level': plev_dim_name}) * factor

    # Adjust geopotential height after applying deltas
    if 'q' in pl_vars:
        logger.debug("Re-adjusting geopotential height with specific humidity")
        z_baro_after = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['q'], sp=True, plev_dim_name=plev_dim_name)
    elif 'r' in pl_vars:
        logger.debug("Re-adjusting geopotential height with relative humidity")
        z_baro_after = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['r'], plev_dim_name=plev_dim_name)

    delta_z = z_baro_before - z_baro_after
    initial_condition_P['z'] = initial_condition_P['z'] - delta_z

    return initial_condition_S, initial_condition_P

def fix_cmip6_data(delta_files, levels):
    """
    Process CMIP6 data by converting pressure levels, interpolating to specified levels, and regridding.

    Parameters:
        delta_files (dict): File paths for delta data per variable.
        levels (list): Pressure levels to interpolate to (in hPa).

    Returns:
        xarray.Dataset: Processed and interpolated CMIP6 dataset.
    """
    logger.info("Fixing CMIP6 data")
    
    # Map delta variable names to CMIP6 variable names
    dict_vars = {'t2m': 'tas', 'tcwv': 'prw', 't': 'ta', 'r': 'hur', 'q': 'hus'}

    # Read and merge variables
    ds_vars = {var: xr.open_dataset(delta_file)[dict_vars[var]] for var, delta_file in delta_files.items()}
    ds_vars = xr.merge(ds_vars.values())
    ds_vars = ds_vars.rename({v: k for k, v in dict_vars.items() if v in ds_vars.data_vars})

    # Convert pressure levels from Pa to hPa
    ds_vars['plev'] = ds_vars['plev'] / 100
    ds_vars['plev'].attrs['units'] = 'hPa'

    # Interpolate to new pressure levels
    interpolated_ds = xr.Dataset()
    for var_name in ds_vars.data_vars:
        if 'plev' in ds_vars[var_name].dims:
            logger.debug(f"Interpolating variable {var_name} to new pressure levels")
            interpolated_var = ds_vars[var_name].interp(plev=levels)
            interpolated_ds[var_name] = interpolated_var
        else:
            interpolated_ds[var_name] = ds_vars[var_name]

    interpolated_ds['plev'] = levels
    interpolated_ds = interpolated_ds.rename({'plev': 'level'})

    # Regrid to 0.25-degree resolution
    new_lons = np.arange(0, 360, 0.25)
    new_lats = np.arange(90, -90.1, -0.25)
    logger.debug("Regridding dataset to 0.25-degree resolution")
    interpolated_grid_ds = interpolated_ds.interp(lon=new_lons, lat=new_lats, method='linear', kwargs={"fill_value": "extrapolate"})
    interpolated_grid_ds = interpolated_grid_ds.rename({'lat': 'latitude', 'lon': 'longitude'})

    return interpolated_grid_ds

def interpolate_to_dayofyear(data, day_of_year, method='linear'):
    """
    Interpolates climatology data to a specific day of the year.

    Parameters:
        data (xarray.Dataset or xarray.DataArray): Monthly climatology dataset or array.
        day_of_year (int): Target day of the year for interpolation.
        method (str): Interpolation method ('linear' by default).

    Returns:
        xarray.Dataset or xarray.DataArray: Interpolated data for the specified day of the year.
    """
    logger.debug(f"Interpolating data to day of year: {day_of_year}")
    
    num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dayofyear = [(sum(num_days[:i]) + num_days[i] // 2 + 1) for i in range(12)]

    data = data.rename({'time': 'dayofyear'})
    data_ext = data.pad(dayofyear=1, mode='wrap')
    dayofyear_ext = [dayofyear[0] - 30] + dayofyear + [dayofyear[-1] + 30]
    data_ext = data_ext.assign_coords({'dayofyear': dayofyear_ext})

    return data_ext.interp(dayofyear=day_of_year, method=method)

def get_model_params(ai_model, path_delta_mm):
    """
    Get delta files and variables based on the AI model.

    Parameters:
        ai_model (str): AI model name ('fcnv2', 'pangu').
        path_delta_mm (str): Base path to the multimodel delta files.

    Returns:
        tuple: (delta_files, surf_vars, pl_vars)
            - delta_files (dict): Dictionary with paths to delta files.
            - surf_vars (list): List of surface variables.
            - pl_vars (list): List of pressure level variables.
    """
    logger.info(f"Getting model parameters for {ai_model}")

    if 'fcnv2' in ai_model:
        delta_files = {
            't2m': path_delta_mm + 'tas/tas_multimodel_mean.nc',
            'tcwv': path_delta_mm + 'prw/prw_multimodel_mean.nc',
            't': path_delta_mm + 'ta/ta_multimodel_mean.nc',
            'r': path_delta_mm + 'hur/hur_multimodel_mean.nc'
        }
        surf_vars = ['t2m', 'tcwv']
        pl_vars = ['t', 'r']
    elif 'pangu' in ai_model:
        delta_files = {
            't2m': path_delta_mm + 'tas/tas_multimodel_mean.nc',
            't': path_delta_mm + 'ta/ta_multimodel_mean.nc',
            'q': path_delta_mm + 'hus/hus_multimodel_mean.nc'
        }
        surf_vars = ['t2m']
        pl_vars = ['t', 'q']
    else:
        logger.error("Unsupported AI model: {}".format(ai_model))
        raise ValueError("Unsupported AI model: {}".format(ai_model))

    return delta_files, surf_vars, pl_vars

# Example usage
if __name__ == "__main__":

    # Generate list of initialization dates
    init_times = []
    current_time = time_i
    while current_time <= time_f:
        init_times.append(current_time)
        current_time += dt.timedelta(hours=delta_h)

    # Retrieve delta files and variables
    delta_files, surf_vars, pl_vars = get_model_params(ai_model, path_delta_mm)
    ds_cmip6_deltas = fix_cmip6_data(delta_files, levels)

    for date in init_times:
        logger.info(f"Processing date: {date}")
        
        # Define file paths for initial conditions
        yyyymmddhh = date.strftime('%Y%m%d%H')
        ending = 'grib' if grib_opt else 'nc'
        initial_condition_files = {
            'surface': path_ic + f'{yyyymmddhh}/{ai_model}_sl_{yyyymmddhh}.{ending}',
            'pressure': path_ic + f'{yyyymmddhh}/{ai_model}_pl_{yyyymmddhh}.{ending}'
        }

        # Interpolate CMIP6 deltas to the day of the year
        day_of_year = date.timetuple().tm_yday
        ds_cmip6_deltas_doy = interpolate_to_dayofyear(ds_cmip6_deltas, day_of_year)

        # Apply deltas to initial conditions
        mod_initial_condition_S, mod_initial_condition_P = apply_delta_to_initial_condition_cmip(
            initial_condition_files, ds_cmip6_deltas_doy, surf_vars, pl_vars, mode='subtract', grib=grib_opt, plev_dim_name=plev_dim_name
        )

        # Create output directory and save modified datasets
        output_path = f'{outputdir}/{yyyymmddhh}'
        os.makedirs(output_path, exist_ok=True)
        logger.debug(f"Saving modified surface dataset to {output_path}/{ai_model}_sl_PGW_multimodel_{yyyymmddhh}.nc")
        mod_initial_condition_S.to_netcdf(f'{output_path}/{ai_model}_sl_PGW_multimodel_{yyyymmddhh}.nc')
        logger.debug(f"Saving modified pressure dataset to {output_path}/{ai_model}_pl_PGW_multimodel_{yyyymmddhh}.nc")
        mod_initial_condition_P.to_netcdf(f'{output_path}/{ai_model}_pl_PGW_multimodel_{yyyymmddhh}.nc')
