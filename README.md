# ARLreader

Python only library to read the NOAA ARLs packed format for HYSPLIT (<https://ready.arl.noaa.gov/HYSPLIT.php>).
Currently only wokring for the GDAS1 assimilation data (<https://www.ready.noaa.gov/gdas1.php>), which is also available from ARL (<ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1>).
A more extensive description of the format is provided in: [Things to know when working with the ARL binary fomat](working_with_ARLformat.md.md)


### Usage

```python
import peakTree
peaks, pt, threslist = peakTree.detect_peak_recursive(data_array, threshold, lambda thres: thres*1.5)
```

Internally following steps are employed:
```python
pt.insert(peak, thres)
pt.concat()
pt.extendedges()
print(pt)
```


### Tests
`python3 -m pytest -v`

### License
Copyright 2017, Martin Radenz
[MIT License](http://www.opensource.org/licenses/mit-license.php)