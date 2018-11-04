#! /usr/bin/env python3
# coding=utf-8

import datetime
import numpy as np
import ARLreader

class TestARLreader():
    def test_filename(self):
        #assert 
        assert ARLreader.fname_from_date(datetime.datetime(2014, 3, 15)) == 'gdas1.mar14.w3'
        assert ARLreader.fname_from_date(datetime.datetime(2014, 5, 7)) == 'gdas1.may14.w1'
        assert ARLreader.fname_from_date(datetime.datetime(2014, 9, 8)) == 'gdas1.sep14.w2'
        assert ARLreader.fname_from_date(datetime.datetime(2014, 1, 28)) == 'gdas1.jan14.w4'
        assert ARLreader.fname_from_date(datetime.datetime(2014, 7, 29)) == 'gdas1.jul14.w5'


    def test_indexinfo(self):
        inp = [14, 4, 1, 0, 0, 0, 99, 'INDX', 0, 0.0, 0.0]
        #result = recordinfo(y=14, m=4, d=1, h=0, fc=0, lvl=0, grid=99, name='INDX', exp=0, prec=0.0, initval=0.0)
        keys =  ['y', 'm', 'd', 'h', 'fc', 'lvl', 'grid', 'name', 'exp', 'prec', 'initval']
        result = ARLreader.convertindexlist(inp)._asdict()
        for i, k in enumerate(keys):
            assert result[k] == inp[i]


    def test_interpolation(self):
        res = ARLreader.interp2d(np.array([[10,20], [5,10]]), np.array([1,2]), np.array([1,2]), (1.5, 1.5))
        np.testing.assert_almost_equal(res, 11.25)

    def test_split_format(self):
        result = ARLreader.split_format('s4,i3,f7', b'ABCD  4.000056')
        assert all([result[0] == 'ABCD', result[1] == 4, result[2] == 0.000056])
