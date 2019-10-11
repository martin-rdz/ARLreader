#! /usr/bin/env python3
# coding=utf-8

import datetime
import ARLreader as Ar

gdas_file = Ar.fname_from_date(datetime.datetime.now())
print('name of input file ', gdas_file)

gdas = Ar.reader('/Users/yinzhenping/Downloads/current7days')
# print('indexinfo ', gdas.indexinfo)
# print('headerinfo ', gdas.headerinfo)
# for i, v in gdas.levels.items():
#     print(i, ' level ', v['level'], list(map(lambda x: x[0], v['vars'])))
# recinfo, grid, data = gdas.load_heightlevel(8, 3, 1, 'RH2M')
# print('recinfo ', recinfo)
# print('grid ', grid)
# print('data ', data)
# recinfo, grid, data = Ar.reader('/Users/yinzhenping/Downloads/current7days').load_heightlevel(8, 3, 850, 'TEMP')

profile, sfcdata, indexinfo, ind = Ar.reader('/Users/yinzhenping/Downloads/current7days').load_profile(11, 3, (51.3, 12.4))
print(profile)
Ar.write_profile('testfile.txt', indexinfo, ind, (51.3, 12.4), profile, sfcdata)