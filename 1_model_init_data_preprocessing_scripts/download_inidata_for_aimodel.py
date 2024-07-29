import cdsapi
import numpy as np
import os
import datetime

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

c = cdsapi.Client()

#Configuration of the download

#modelname = 'fcnv2'
modelname = 'pangu'
if modelname == 'fcnv2':
    plevels = [ '50', '100', '150', '200', '250', '300', '400','500', '600', '700', '850', '925', '1000']
    variables_pl = [ '129', '130', '131', '132', '157'] #
    variables_sf = [ '134', '137', '151', '165', '166', '167', '228246', '228247']
elif modelname == 'pangu':
    plevels = [ '50', '100', '150', '200', '250', '300', '400','500', '600', '700', '850', '925', '1000']
    variables_pl = [ '129', '130', '131', '132', '133'] #z, t, u, v, q
    variables_sf = ['151', '165', '166', '167'] #msl, u10m, v10m, rt2m

resolution = ['0.25', '0.25'] # cannot be changed for monthly means as far as I know

first_init_time = datetime.datetime(2018,8,8,18)
end_init_time = datetime.datetime(2018,8,8,18)
delta_hours = 6

init_times = []
current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)

#add a second period
first_init_time = datetime.datetime(2018,9,5,00)
end_init_time = datetime.datetime(2018,9,15,18)
current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)

#third period
first_init_time = datetime.datetime(2023,10,25,0)
end_init_time = datetime.datetime(2023,11,5,18)
current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)

def download_init(yyyymmddhh, modelname, leveltype, variables, dataset, plevels, savedir):
    try:
        filename = f'{modelname}_{leveltype}_{yyyymmddhh}.grib'
        if os.path.exists(os.path.join(savedir, filename)):
            print(f'{filename} already exists. Skipping download.')
            return
        print(f'Downloading init {yyyymmddhh} for {modelname} {leveltype}')

        year = yyyymmddhh[0:4]
        month = yyyymmddhh[4:6]
        day = yyyymmddhh[6:8]
        hour = yyyymmddhh[8:10]
        c.retrieve(
            dataset,
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': variables,
                'pressure_level': plevels,
                'year': year,
                'month' : month,
                'day' : day,
                'time': f'{hour}:00',
            },
            os.path.join(savedir, filename)
        )
        print(f'Download complete for .')
    except Exception as e:
        print(f"Error during download: {e}")

# Use ThreadPoolExecutor for parallel downloads
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for init in init_times:
        print(init)

        yyyymmddhh = init.strftime('%Y%m%d%H')
        savedir = f'/pool/usuarios/bernatj/Data/ai-forecasts/input/grib/{yyyymmddhh}'
        # Create the directory (and its parent directories if missing)
        os.makedirs(savedir, exist_ok=True)

        futures.append(executor.submit(download_init,yyyymmddhh, modelname, 'sl', variables_sf, 'reanalysis-era5-single-levels', plevels, savedir))
        futures.append(executor.submit(download_init,yyyymmddhh, modelname, 'pl', variables_pl, 'reanalysis-era5-pressure-levels', plevels, savedir))

    # Wait for all downloads to finish
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error during download: {e}")
