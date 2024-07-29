import cdsapi
import os
import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Initialize the CDS API client
c = cdsapi.Client()

# =============================
# User Configuration Section
# =============================

# Model to use: 'fcnv2' or 'pangu'
modelname = 'pangu'  # Change to 'fcnv2' if using the FourCastNet model

# Define pressure levels and variables based on the selected model
if modelname == 'fcnv2':
    plevels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
    variables_pl = ['129', '130', '131', '132', '157']  # Geopotential, Temperature, U/V wind components, Relative humidity
    variables_sf = ['134', '137', '151', '165', '166', '167', '228246', '228247']  # Surface variables
elif modelname == 'pangu':
    plevels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
    variables_pl = ['129', '130', '131', '132', '133']  # Geopotential, Temperature, U/V wind components, Specific humidity
    variables_sf = ['151', '165', '166', '167']  # Mean sea level pressure, U/V wind components at 10m, 2m temperature

# Resolution setting (fixed for ERA5 monthly means)
resolution = ['0.25', '0.25']

# Define periods to download data for (modify as needed)
time_periods = [
    (datetime.datetime(2018, 7, 22, 0), datetime.datetime(2018, 8, 8, 18)),
    (datetime.datetime(2018, 9, 5, 0), datetime.datetime(2018, 9, 15, 18)),
    (datetime.datetime(2023, 10, 25, 0), datetime.datetime(2023, 11, 5, 18))
]
delta_hours = 6  # Time interval between data downloads

# Directory to save downloaded data
base_savedir = '/pool/usuarios/bernatj/Data/ai-forecasts/input/grib'

# Maximum number of parallel downloads
max_workers = 10

# =============================
# End of User Configuration
# =============================

# Generate initialization times for each period
init_times = []
for first_init_time, end_init_time in time_periods:
    current_time = first_init_time
    while current_time <= end_init_time:
        init_times.append(current_time)
        current_time += datetime.timedelta(hours=delta_hours)

def download_init(yyyymmddhh, modelname, leveltype, variables, dataset, plevels, savedir):
    """
    Downloads weather data for a specific initialization time.

    Args:
        yyyymmddhh (str): The initialization time in 'YYYYMMDDHH' format.
        modelname (str): The name of the model ('fcnv2' or 'pangu').
        leveltype (str): The type of data ('sl' for single levels, 'pl' for pressure levels).
        variables (list): List of variable codes to download.
        dataset (str): The dataset to download from (e.g., 'reanalysis-era5-single-levels').
        plevels (list): Pressure levels to include if applicable.
        savedir (str): Directory to save the downloaded data.

    Returns:
        None
    """
    try:
        filename = f'{modelname}_{leveltype}_{yyyymmddhh}.grib'
        filepath = os.path.join(savedir, filename)
        if os.path.exists(filepath):
            print(f'{filename} already exists. Skipping download.')
            return

        print(f'Downloading init {yyyymmddhh} for {modelname} {leveltype}')
        year, month, day, hour = yyyymmddhh[:4], yyyymmddhh[4:6], yyyymmddhh[6:8], yyyymmddhh[8:10]
        c.retrieve(
            dataset,
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': variables,
                'pressure_level': plevels if leveltype == 'pl' else None,
                'year': year,
                'month': month,
                'day': day,
                'time': f'{hour}:00',
            },
            filepath
        )
        print(f'Download complete for {filename}.')
    except Exception as e:
        print(f"Error during download of {yyyymmddhh} for {modelname} {leveltype}: {e}")

# Use ThreadPoolExecutor for parallel downloads
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for init in init_times:
        yyyymmddhh = init.strftime('%Y%m%d%H')
        savedir = os.path.join(base_savedir, yyyymmddhh)
        os.makedirs(savedir, exist_ok=True)  # Ensure the directory exists

        futures.append(executor.submit(download_init, yyyymmddhh, modelname, 'sl', variables_sf, 'reanalysis-era5-single-levels', plevels, savedir))
        futures.append(executor.submit(download_init, yyyymmddhh, modelname, 'pl', variables_pl, 'reanalysis-era5-pressure-levels', plevels, savedir))

    # Wait for all downloads to finish
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error during download: {e}")
