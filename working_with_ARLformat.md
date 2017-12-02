# Things to know when working with the ARL binary fomat

One very convenitent way to obtain near-realtime gridded meteorological analyses are the GDAS files provided by NOAA ARL for their HYSPLIT trajectory model (<https://www.arl.noaa.gov/hysplit/hysplit/>). The server can be found here: <ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1>
This source hast two major advantages:
1. very small files (600MB/week at a resolution of 1deg and 3h)
2. data since Dec 2004

It is no quality checked reanalysis, but can be quite handy for first overview or a coarse analysis.
One major downside is the dataformat. It's a custom binary format neither compatible with grib nor netcdf.
A rough description is available at <https://www.ready.noaa.gov/gdas1.php> but it might not answer all questions regarding the format.
The archive is split up into weekly files (w1 = days 1-7, w2 = days 8-14, w3 = days 15-21, w4 = days 22-28, w5 = rest of month).
The grid is based on 1deg latitude and longitude (360*181 points) and starts at the lower left corner (0W, 90S) and runs to (1W, 90W).
Each file can be split up into records of 65210 bytes - a header of 50 bytes and a one byte entry per gridpoint (360*181). The different parameters at every height level and all height levels are simply chained together.
The first record at every timestep contains a 50 byte header, followed by the grid definition and a list of all variables encoded in ASCII.
Usually this record contains 1654 bytes of data. The remainder (63556 bytes) of this record is empty. 
The structure of the ASCII header is:

| Field | Format | Description  |
| --- | --- | --- |
| Month | I2 | Greenwich date for which data valid  |
| Day | I2 | " |
| Hour | I2 | " |
| Forecast | I2 | forecast, zero for analysis |
| Level | I2 | Level from the surface up |
| Grid | I2 |  Grid identification |
| Variable | A4 | Variable label |
| Exponent | I4 | Scaling exponent needed for unpacking |
| Precision | E14.7 | Precision of unpacked data |
| Value 1,1 | E14.7 | Unpacked data value at grid point 1,1 |

and the grid definition:
| Format | Information |
| --- | --- | 
| A4 | Data Source | 
| I3 | Forecast hour |
| I2 | Minutes associated with data time |
| 12F7. | 1) Pole Lat, 2) Pole Long, 3) Tangent Lat, 4) Tangent Long, 5) Grid Size, 6) Orientation, 7) Cone Angle, 8) X-Synch pnt, 9) Y-Synch pnt, 10) Synch pnt lat, 11) Synch pnt long, 12) Reserved |
| 3I3 | 1) Numb x pnts, 2) Numb y pnts, 3) Numb levels |
| I2 | Vertical coordinate system flag | 
| I4 | Length in bytes of the index record, excluding the first 50 bytes
| | LOOP: number of data levels |
| F6. | height of the first level |
| I2 | number of variables at that level |
| | LOOP: number of variables |
| A4 | variable identification | 
| I3 | rotating checksum of the packed data | 
| 1X | Reserved space for future use |

The following record contains the first parameter preceded by the 50 byte header.
The values are packed by calculating the difference to the preciding value and scaling the result between 0 and 254.
The inital value and the scaling factor are stored in the header. The order is not strictly linear, the first column has to be decoded first:

```python
value_old = data[0,j]
for i in range(1,nx):
    ri = j*nx + i
    val = (binarray[ri]-127)/(2**(7-exp)) + value_old
    value_old = val
    if abs(val) < prec:
        val = 0.
    data[i,j] = val
    if stopat != None:
        if i >= stopat[0] and j >= stopat[1]:
            break
```


### Example (single record are denoted by sqare brackets)
```
first timestep:
[header (50 bytes), grid and variables (1604 bytes)]
[header (50 bytes), parameter (surface)] 
[header (50 bytes), parameter (surface)] 
...
[header (50 bytes), parameter(1000hPa)]
...
...
[header (50 bytes), parameter (700hPa)]
...
second timestep:
[header (50 bytes), grid and variables (1604 bytes)]
[header (50 bytes), parameter (surface)] 
[header (50 bytes), parameter (surface)] 
...
[header (50 bytes), parameter(1000hPa)]
...
...
[header (50 bytes), parameter (700hPa)]
...
```