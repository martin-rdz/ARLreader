# ARLreader

Python only library to read the NOAA ARLs packed format for HYSPLIT (<https://ready.arl.noaa.gov/HYSPLIT.php>).
Currently only wokring for the GDAS1 assimilation data (<https://www.ready.noaa.gov/gdas1.php>), which is also available from ARL (<ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1>).
A more extensive description of the format is provided in: [Things to know when working with the ARL binary fomat](working_with_ARLformat.md)

Currently only GDAS1 and the profiles of GDAS0p5 and GDAS0p25 are working.

### Requirements

Either python3 or [`Anaconda3`](https://www.anaconda.com/distribution/) with the libraries specified in [requirements.txt](requirements.txt).

### Installation 

**Installation within a new conda virtual environment**

create a new virtual environment (here for anaconda)

```bash
conda create -n ARLreader   # you can choose other names as well, 
                            # but using the consistent name during the installation.
activate ARLreader   # activate the virtual environement

conda install python=3.6   # install python, shipped with `pip`, `setuptools`...
git clone https://github.com/martin-181/ARLreader.git   # download the code repository
cd ARLreader
pip install -r requirements.txt   # install the dependencies for ARLreader
```

compile the fast reading functions (optional)

```bash
python setup.py install
```



### Usage

#### python interface

Reading a 2d Field:
```python
import ARLreader as Ar

gdas = Ar.reader('data/gdas1.apr14.w1')
print('indexinfo ', gdas.indexinfo)
print('headerinfo ', gdas.headerinfo)
for i, v in gdas.levels.items():
    print(i, ' level ', v['level'], list(map(lambda x: x[0], v['vars'])))
# load_heightlevel(day, houer, level, variable)
recinfo, grid, data = gdas.load_heightlevel(2, 3, 0, 'RH2M')
```

Read the profile at a given location an write it to a text file with `load_profile(day, hour, (lat, lon))`:
```python
profile, sfcdata, indexinfo, ind = Ar.reader('data/gdas1.apr14.w1').load_profile(2, 3, (51.3, 12.4))
print(profile)
Ar.write_profile('testfile.txt', indexinfo, ind, (51.3, 12.4), profile, sfcdata)
```

Get the filename from a datetime `Ar.fname_from_date(datetime.datetime(2014, 4, 3))`.


![example](img/comparison_GDAS_radiosonde.png)



#### command line interface

```text
ARLreader -h   # prompt up the help
```

Below is the help messages for using `ARLreader`:
```text
usage: ARLreader [-h] [-s START_TIME] [-e END_TIME] [--latitude LATITUDE]
                 [--longitude LONGITUDE] [--station STATION]
                 [-f GLOBAL_FOLDER] [-o PROFILE_FOLDER] [--force]

extract the GDAS1 profile from GDAS1 global binary data.

optional arguments:
  -h, --help            show this help message and exit
  -s START_TIME, --start_time START_TIME
                        start time of your interests (yyyymmdd-HHMMSS).
                        e.g.20151010-120000
  -e END_TIME, --end_time END_TIME
                        stop time of your interests (yyyymmdd-HHMMSS).
                        e.g.20151010-120000
  --latitude LATITUDE   latitude of your station. (-90, 90).
                        Default: 30.533721
  --longitude LONGITUDE
                        longitude of your station. (0, 360).
                        Default: 114.367216
  --station STATION     station name.
                        Default: wuhan
  -f GLOBAL_FOLDER, --global_folder GLOBAL_FOLDER
                        folder for saving the GDAS1 global files.
                        e.g., 'C:\Users\zhenping\Desktop\global'
  -o PROFILE_FOLDER, --profile_folder PROFILE_FOLDER
                        folder for saving the extracted profiles
                        e.g., 'C:\Users\zhenping\Desktop\wuhan'
  --force               force to download the GDAS1 global dataset (not suggested)
```

**setup the reader for a new station**

```
ARLreader -s 20190920-000000 -e 20190923-000000 --latitude 51.35 --longitude 12.35 --station leipzig -f <data_folder> -o <output_folder>
```

### Tests
`python3 -m pytest -v`

### License
The code is partly based on a prior implementation in IDL by Patric Seifert.

Copyright 2017, Martin Radenz, Yin Zhenping
[MIT License](http://www.opensource.org/licenses/mit-license.php)
