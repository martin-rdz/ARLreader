# ARLreader

Python only library to read the NOAA ARLs packed format for HYSPLIT (<https://ready.arl.noaa.gov/HYSPLIT.php>).
Currently only wokring for the GDAS1 assimilation data (<https://www.ready.noaa.gov/gdas1.php>), which is also available from ARL (<ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1>).
A more extensive description of the format is provided in: [Things to know when working with the ARL binary fomat](working_with_ARLformat.md)


### Usage

Reading a 2d Field:
```python
import ARLreader as Ar
recinfo, grid, data = Ar.reader('data/gdas1.apr14.w1').load_heightlevel(2, 3, 0, 'RH2M')
```

Read the profile at a given location an write it to a text file:
```python
profile, sfcdata, indexinfo, ind = Ar.reader('data/gdas1.apr14.w1').load_profile(2, 3, (51.3, 12.4))
print(profile)
Ar.write_profile('testfile.txt', indexinfo, ind, (51.3, 12.4), profile, sfcdata)
```

Get the filename from a datetime `Ar.fname_from_date(datetime.datetime(2014, 4, 3))`.

### Tests
`python3 -m pytest -v`

### License
Copyright 2017, Martin Radenz
[MIT License](http://www.opensource.org/licenses/mit-license.php)