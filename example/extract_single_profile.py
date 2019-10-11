#! /usr/bin/env python3
# coding=utf-8

import datetime
import ARLreader as Ar

gdas_file = Ar.fname_from_date(datetime.datetime(2014, 4, 3))
print('name of input file ', gdas_file)

gdas = Ar.reader('data/gdas1.apr18.w1')
print('indexinfo ', gdas.indexinfo)
print('headerinfo ', gdas.headerinfo)
for i, v in gdas.levels.items():
    print(i, ' level ', v['level'], list(map(lambda x: x[0], v['vars'])))
# load_heightlevel(day, houer, level, variable)
recinfo, grid, data = gdas.load_heightlevel(2, 3, 0, 'RH2M')
print('recinfo ', recinfo)
print('grid ', grid)
print('data ', data)
recinfo, grid, data = Ar.reader('data/gdas1.apr14.w1').load_heightlevel(2, 3, 850, 'TEMP')

profile, sfcdata, indexinfo, ind = Ar.reader('data/gdas1.apr14.w1').load_profile(2, 3, (51.3, 12.4))
print(profile)
Ar.write_profile('testfile.txt', indexinfo, ind, (51.3, 12.4), profile, sfcdata)