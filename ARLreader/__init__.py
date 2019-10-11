#! /usr/bin/env python3
# coding=utf-8

"""
ARLreader
Author radenz@tropos.de

Copyright 2017, Martin Radenz, [MIT License]
"""

import numpy as np
import struct
import collections
import datetime
import sys
import os
import logging
from ftplib import FTP
import argparse
from argparse import RawTextHelpFormatter


# Constants Definition
gridtup = collections.namedtuple('gridtup', 'lats, lons')
FTPHost = 'arlftp.arlhq.noaa.gov'   # ftp link of the ARL server
FTPFolder = 'pub/archives/gdas1'   # folder for the GDAS1 data
LOG_MODE = 'DEBUG'
LOGFILE = 'log'
PROJECTDIR = os.path.dirname(os.path.realpath(__file__))


def logger_def():
    """
    initialize the logger
    """

    logFile = os.path.join(PROJECTDIR, LOGFILE)
    logModeDict = {
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'DEBUG': logging.DEBUG,
        'ERROR': logging.ERROR
        }
    logger = logging.getLogger(__name__)
    logger.setLevel(logModeDict[LOG_MODE])

    fh = logging.FileHandler(logFile)
    fh.setLevel(logModeDict[LOG_MODE])
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logModeDict[LOG_MODE])

    formatterFh = logging.Formatter('%(asctime)s - %(name)s - ' +
                                    '%(levelname)s - %(funcName)s - ' +
                                    '%(lineno)d - %(message)s')
    formatterCh = logging.Formatter(
        '%(message)s')
    fh.setFormatter(formatterFh)
    ch.setFormatter(formatterCh)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# initialize the logger
logger = logger_def()

try:
    # compile and load the c-version
    # import pyximport
    # pyximport.install()
    import ARLreader.fast_funcs as fast_funcs
    fastfuncsavail = True
    logger.info("fast_funcs available")
except Exception as e:
    # use the numpy only version
    fastfuncsavail = False
    logger.info("fast_funcs not available")


def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis
    Args:
        array (np.array): sorted array with values
        value: value to find
    Returns:
        index: integer
    """
    i = np.searchsorted(array, value) - 1
    if not i == array.shape[0] - 1 \
            and np.abs(array[i] - value) > np.abs(array[i + 1] - value):
        i = i + 1
    return i


def vapor_press(p, mr):
    '''
    Args:
        p: The path of the file to wrap
        mr: mixing ratio

    Returns:
        water vapor partial pressure (same unit as p)
    '''
    return p * mr / (0.621980 + mr)


def dewpoint(e):
    '''
    Args:
        e: water vapor partial pressure

    Returns:
        dewpoint temperature wrt to liquid water
    '''
    val = np.log(e / 6.112)
    return (243.5 * val) / (17.67 - val)


def rh_from_q(p, q, tK):
    '''
    Args:
        p: pressure
        q: specific humidity
        tK: temperature [K]

    Returns:
        relative humidity
    '''
    e_saet = 6.112 * np.exp(17.67 * (tK - 273.15) / (tK - 29.65))
    wsaet = 0.621980 * e_saet / (p - e_saet)
    rh = 100 * q/((1-q)*wsaet)
    return rh


def rh_to_q(rh, tK, p):
    '''
    Args:
        rh: relative humidity [%]
        tK: temperature [K]
        p: pressure [hPa]

    Returns:
        specific humidity [kg/kg]
    '''
    e_saet = 6.112 * np.exp(17.67 * (tK - 273.15) / (tK - 29.65))
    e = (rh/100.)*e_saet
    return (0.622*e) / (p - 0.78*e)


def equipottemp(p, q, tK):
    '''
    Args:
        p: pressure [hPa]
        q: specific humidity [kg/kg]
        tK: temperature [K]

    Returns:
        equipotential temperature
    '''
    mr = q/(1.-q)
    e = vapor_press(p, mr)
    td = dewpoint(e)
    td += 273.15
    kappa = 0.2854
    # convert to g/kg
    mr *= 1e3
    t_l = 56 + 1. / (1. / (td - 56) + np.log(tK / td0) / 800.)
    th_l = tK * (1000. / (p-e)) ** kappa * (tK / t_l) ** (0.28e-3 * mr)
    th_e = th_l * np.exp((3.036 / t_l - 0.00178) * mr * (1 + 0.448e-3 * mr))
    return th_e


def pottemp(tK, p):
    '''
    Args:
        tK: temperature [K]
        p: pressure [hPa]

    Returns:
        potential temperature
    '''
    return tK*(1000./p)**(0.286)


def wind_from_components(u, v):
    '''
    Args:
        u: zonal wind component
        v: meridional wind component

    Returns:
        wind direction, wind velocity
    '''
    assert all([isinstance(u, np.ndarray), isinstance(v, np.ndarray)]), \
        'input needs to be numpy array'
    wdir = 90. - (np.arctan2(-v, -u)*180./np.pi)
    wdir[wdir < 0] += 360.
    wvel = np.sqrt(u * u + v * v)
    return wdir, wvel


def fname_from_date(dt):
    '''
    Retrieve the respective GDAS1 file, based on the input timestamp.

    Parameters
    ----------
    dt (datetime): date

    Returns
    -------
    name for the gdas file, eg "gdas1.oct14.w1" or "current7days"
    w1 = 1-7 (days)
    w2 = 8-14
    w3 = 15-21
    w4 = 22-28
    w5 = others
    '''

    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
              7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    week_no = ((dt.day - 1) // 7) + 1

    # determine the current 7 days
    currentday_start = (week_no - 1) * 7 + 1
    currentDate = datetime.datetime.now()
    currentDate_weekstart = datetime.datetime(
        currentDate.year,
        currentDate.month,
        currentday_start
    )
    if (dt >= currentDate_weekstart) and (dt <= currentDate):
        gdas1File = 'current7days'
    elif (dt > currentDate):
        logger.info('GDAS1 file for input date is not ready yet.')
        raise FileNotFoundError
    elif (dt < currentDate_weekstart):
        gdas1File = 'gdas1.{}{}.w{}'.format(
            months[dt.month],
            dt.strftime('%y'),
            week_no
            )

    return gdas1File


def split_format(fmtstring, s):
    """decode a bytestring and split and convert it by a given format code
    s: string, i: int, f: float, x: none

    Args:
        fmtstring: specifies the format of the string, eg 's4,i4,f14,f14'
        s: string to decode

    Returns:
        list: with results
    """
    decoders = {'s': str, 'i': int, 'f': float, 'x': lambda x: ''}
    s = s.decode(encoding='ascii')
    result = []
    for fmt in fmtstring.split(','):
        # print('fmt code ', fmt)
        result.append(decoders[fmt[0]](s[:int(fmt[1:])]))
        s = s[int(fmt[1:]):]
    return list(filter(lambda x: x != '', result))


def convertindexlist(l):
    """convert the list of decoded ASCII headerinformation to a namedtupel

    Args:
        l (list): decoded header information

    Returns:
        namedtupel: headerinfo
    """
    recordinfo = collections.namedtuple('recordinfo',
                                        'y m d h fc lvl grid name exp ' +
                                        'prec initval')
    return recordinfo(y=l[0], m=l[1], d=l[2], h=l[3],
                      fc=l[4], lvl=l[5], grid=l[6], name=l[7],
                      exp=l[8], prec=l[9], initval=l[10])


def calc_no_within_ts(levels, lvl, varname):
    """
    calculate the position for a given level and variable inside the
    current timestep

    Args:
        levels: all levels
        lvl: the level we are interested in
        varname: the variable we are interested in

    Returns:
        no of steps
    """
    steps = 0
    for i in range(len(levels)):
        if lvl == levels[i]['level']:
            steps += list(map(lambda x: x[0], levels[i]['vars'])).index(
                varname
                )
            break
        steps += len(levels[i]['vars'])
    return steps


def get_lvl_index(levels, lvl):
    """index of a given level in levels"""
    for k, v in levels.items():
        if v['level'] == lvl:
            return k, v


def calc_index(indexinfo, headerinfo, levels, day, hour, lvl, varname):
    """
    calculate the starting byte for a given time, level and variable

    Args:
        indexinfo: for the binary file
        headerinfo: for the binary file
        levels: all levels
        day (int): selected day
        hour (int): selected hour
        lvl: selected level
        varname: the variable we are interested in

    Returns:
        starting byte
    """

    rec_length = (headerinfo['Nx'] * headerinfo['Ny']) + 50
    no_vars = sum([len(v['vars']) for v in levels.values()])
    # important for each timestep you have to jump a whole record
    # (index + header + variables (+empty) are replicated)
    timesteplength = (rec_length * no_vars) + rec_length
    no_full_slices = ((day - indexinfo.d) * 8 + int(hour / 3)) * timesteplength
    no_at_timestep = calc_no_within_ts(levels, lvl, varname) * rec_length
    # plus the initial record length
    assert no_full_slices + no_at_timestep + rec_length > 0, \
        'calculated negative index, check your inputs'

    return no_full_slices + no_at_timestep + rec_length


def read_data(fname, bin_index, headerinfo, stopat=(-1, -1)):
    """
    read the data from a given file based on the starting index and decode the
    data with unpack_data()

    Args:
        fname: filename
        bin_index: starting bin
        headerinfo: for the binary file
        stopat (optional, int): stop at bin no

    Returns:
        recordinfo, data
    """

    rec_length = (headerinfo['Nx']*headerinfo['Ny'])+50
    with open(fname, mode='rb') as f:
        f.seek(bin_index, 0)
        unpacked = b''.join(struct.unpack(50*'s', f.read(50)))
        recinfo = convertindexlist(split_format(
            6 * 'i2,'+'s2,s4,i4,f14,f14',
            unpacked
            ))
        unpacked = struct.unpack((rec_length-50) * 'B', f.read(rec_length-50))
        if fastfuncsavail:
            data = fast_funcs.unpack_data(
                np.array(unpacked),
                headerinfo['Nx'],
                headerinfo['Ny'],
                recinfo.initval,
                recinfo.exp,
                recinfo.prec,
                stopat=stopat
                )
        else:
            data = unpack_data(
                np.array(unpacked),
                headerinfo['Nx'],
                headerinfo['Ny'],
                recinfo.initval,
                recinfo.exp,
                recinfo.prec,
                stopat=stopat
                )

    return recinfo, data


def unpack_data(binarray, nx, ny, initval, exp, prec, stopat=(-1, -1)):
    """
    unpack the binary data with the option to stop at a given index
    """

    data = np.zeros((nx, ny))
    value_old = initval

    if stopat != (-1, -1):
        nx_min = max(stopat[1] - 1, 0)
        nx_max = min(stopat[1] + 1, nx)
        ny_min = max(stopat[0] - 1, 0)
        ny_max = min(stopat[0] + 1, ny)
    else:
        nx_min = 0
        nx_max = nx
        ny_min = 0
        ny_max = ny

    for j in range(ny_max):
        ri = j*nx
        data[0, j] = ((binarray[ri] - 127) / (2**float(7-exp))) + value_old

        value_old = data[0, j]

    for j in range(ny_min, ny_max):
        value_old = data[0, j]
        for i in range(1, nx_max):
            ri = j*nx + i
            val = (binarray[ri] - 127) / (2**float(7-exp)) + value_old
            value_old = val
            if abs(val) < prec:
                val = 0.
            data[i, j] = val
            if stopat != (-1, -1):
                if i >= stopat[0] and j >= stopat[1]:
                    break
    return data


def calc_p_from_sigma(sigma, p_sfc):
    """
    calculate a levels pressure from sigma coordinate and ground pressure

    Example from
    https://ready.arl.noaa.gov/data/archives/gdas0p5/readme_gdas0p5_info.txt:

    The pressure level is defined as: P = A + .B*PRSS,
    where PRSS is the surface pressure and A.B is defined as the level
    (ie., 16.853 = 869 hPa when the surface pressure is 1000 hPa
    [P=16+0.853*1000])
    """

    A = np.floor(sigma)
    B = sigma - A
    logger.info("A {}".format(A))
    logger.info("B {}".format(B))
    logger.info('pressure {}'.format(A+(B*p_sfc)))

    return A + (B * p_sfc)


def interp(data, lats, lons, coord):
    return interp2d(data, lons, lats, (coord[1], coord[0]))


def interpx(arr, grid, coord):
    return np.transpose(
        (coord - grid[0]) * (arr[:, 1] - arr[:, 0]) /
        (grid[1] - grid[0]) + arr[:, 0]
        )


def interp2d(arr, grid0, grid1, coord):
    """2d interpolation of the values between the grid cells"""
    val = interpx(arr, grid0, coord[0])
    return (coord[1] - grid1[0]) * (val[1] - val[0]) / \
           (grid1[1] - grid1[0]) + val[0]


def write_profile(fname, headerinfo, ind, coord, profile, sfcdata):
    """
    write the profile to a file

    Args:
        fname: filename
        headerinfo: headerinfo
        ind: index of grid point
        coord: coordinates of grid point
        profile: profile with variables
        sfcdata: variables at surface
    """

    potT = pottemp(profile['TEMP'], profile['PRSS'])
    wdir, wvel = wind_from_components(profile['UWND'], profile['VWND'])
    with open(fname, 'w') as f:
        f.write(' Profile Time:  {:2d} {:2d} {:2d} {:2d}  0\n'.format(
            headerinfo.y,
            headerinfo.m,
            headerinfo.d,
            headerinfo.h
            ))

        f.write(' Used Nearest Grid Point ( {:3d}, {:3d}) to Lat:   {:5.2f}, Lon:    {:5.2f}\n'.format(
                    ind[1],
                    ind[0],
                    coord[0],
                    coord[1]
                    ))
        f.write('        2D Fields \n')
        f.write('\n\n\n')
        f.write('        3D Fields \n')
        f.write('        HGTS  TEMP  UWND  VWND  WWND  RELH     ' +
                'TPOT  WDIR  WSPD\n')
        f.write('           m    oC   m/s   m/s   hPa     %     ' +
                '  oK   deg   m/s\n')
        for i in range(profile['VWND'].shape[0]):
            f.write('  {:4.0f} {:5.0f} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}    {:5.1f} {:5.1f}  {:4.1f}\n'.format(
                        profile['PRSS'][i],
                        profile['HGTS'][i],
                        profile['TEMP'][i] - 273.15,
                        profile['UWND'][i],
                        profile['VWND'][i],
                        profile['WWND'][i]*3600,
                        profile['RELH'][i],
                        potT[i],
                        wdir[i],
                        wvel[i]))


def write_profile_plain(fname, headerinfo, ind, coord, profile, sfcdata):
    """
    write the profile to a file

    Args:
        fname: filename
        headerinfo: headerinfo
        ind: index of grid point
        coord: coordinates of grid point
        profile: profile with variables
        sfcdata: variables at surface
    """
    potT = pottemp(profile['TEMP'], profile['PRSS'])
    wdir, wvel = wind_from_components(profile['UWND'], profile['VWND'])
    with open(fname, 'w') as f:
        f.write('# Profile Time:  {:2d} {:2d} {:2d} {:2d}  0\n'
                .format(
                    headerinfo.y,
                    headerinfo.m,
                    headerinfo.d,
                    headerinfo.h
                    ))
        f.write('# Used Nearest Grid Point ( {:3d}, {:3d}) to Lat:   {:5.2f}, Lon:    {:5.2f}\n'.format(
                    ind[1],
                    ind[0],
                    coord[0],
                    coord[1]
                    ))
        f.write('# PRESS HGTS  TEMP  UWND  VWND  WWND  RELH     ' +
                'TPOT  WDIR  WSPD\n')
        f.write('#   hPa    m    oC   m/s   m/s   hPa     %     ' +
                '  oK   deg   m/s\n')
        for i in range(profile['VWND'].shape[0]):
            f.write('  {:4.0f} {:5.0f} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}    {:5.1f} {:5.1f}  {:4.1f}\n'.format(
                            profile['PRSS'][i],
                            profile['HGTS'][i],
                            profile['TEMP'][i] - 273.15,
                            profile['UWND'][i],
                            profile['VWND'][i],
                            profile['WWND'][i] * 3600,
                            profile['RELH'][i],
                            potT[i],
                            wdir[i],
                            wvel[i]
                            ))


class reader():
    """
    read the header of the gdas file
    50 index bytes, 108 grid bytes and the vars at different heights

    Args:
        fname: filename
    """

    def __init__(self, fname):
        self.fname = fname
        with open(fname, mode='rb') as f:
            content = f.read(5000)
            # for python struct s,c,p are chars
            unpacked = b''.join(struct.unpack(50*'s', content[:50]))
            index_info = split_format(6*'i2,'+'s2,s4,i4,f14,f14', unpacked)
            indexinfo = convertindexlist(index_info)
            logger.info('indexinfo '.format(' '.join(map(str, indexinfo))))
            self.indexinfo = indexinfo
            unpacked = b''.join(struct.unpack_from(108*"s", content[:], 50))
            header_format = 's4,i3,i2,'+12*'f7,'+3*'i3,'+'i2,i4'
            header = split_format(header_format, unpacked)
            headerinfo = {'source': header[0], 'fcth': header[1],
                          'minDatatime': header[2], 'griddef': header[3:15],
                          'Nx': header[15], 'Ny': header[16], 'Nz': header[17],
                          'Coordzflag': header[18], 'headerlength': header[19]}
            logger.info('raw header {}'.format(' '.join(map(str, header))))
            logger.info('headerinfo {}'.format(' '.join(map(str, headerinfo))))
            assert headerinfo['source'] in ['GDAS', 'GFSG', 'GFSQ'], \
                'other sources than gdas not supported yet'
            self.headerinfo = headerinfo
            if headerinfo['Nx'] == 720 and headerinfo['Ny'] == 361:
                self.resolution = 0.5
            elif headerinfo['Nx'] == 360 and headerinfo['Ny'] == 181:
                self.resolution = 1.0
            # actual 1440 grid points
            elif headerinfo['Nx'] == 440 and headerinfo['Ny'] == 721:
                self.resolution = 0.25
                headerinfo['Nx'] = 1440
            else:
                raise ValueError("could not infer GDAS resolution")

            # read the variables in the different levels
            levels = {}
            cur = 158
            for ih in range(headerinfo['Nz']):
                # unpack the varinfos
                unpacked = b''.join(struct.unpack_from(
                    8*"s",
                    content[:],
                    cur
                    ))
                height_lvl, Nvars = split_format('f6,i2', unpacked)
                cur += 8
                levels[ih] = {'level': height_lvl, 'vars': []}
                # go through all vars at this height level
                for ivar in range(Nvars):
                    unpacked = b''.join(struct.unpack_from(
                        8 * "s",
                        content[:],
                        cur
                        ))
                    cur += 8
                    varname, checksum = split_format('s4,i3,x1', unpacked)
                    levels[ih]['vars'].append((varname, checksum))
            self.levels = levels

            self.grid = gridtup(lats=np.linspace(-90, 90, headerinfo['Ny']),
                                lons=np.linspace(0, 359, headerinfo['Nx']))

    def load_heightlevel(self, day, hour, level, variable, truelon=True):
        """
        return the full field (recinfo, self.grid, data) for a given level and
        variable

        Args:
            day (int): selected day
            hour (int): selected hour
            level: selected level
            variable: selected variable
            truelon (optional, bool): convert to true  lon

        Returns:
            recordinfo, grid, data
        """

        assert hour % 3 == 0, \
            'Other time resolution than 3h not supported yet.'

        logger.info('{} {}'.format(self.levels, level))
        varlist_at_level = list(map(
            lambda x: x[0],
            get_lvl_index(self.levels, level)[1]['vars']
            ))
        assert variable in varlist_at_level,\
            'Variable not available at this level. Only: ' + \
            str(varlist_at_level)

        bin_index = calc_index(
            self.indexinfo,
            self.headerinfo,
            self.levels,
            day,
            hour,
            level,
            variable
            )

        logger.info("bin index {}".format(bin_index))
        recinfo, data = read_data(self.fname, bin_index, self.headerinfo)
        logger.info('recordinfo {}'.format(recinfo))
        assert all(
            [recinfo.name == variable, recinfo.d == day, recinfo.h == hour]
            ), 'Something went wrong while reading file'

        grid = self.grid
        if truelon:
            logger.info('convert to the true longitude coordinates')
            data = np.roll(data, -181, axis=0)
            lons = np.roll(self.grid.lons, -181, axis=0)
            lons[lons > 180] -= 360.
            grid = gridtup(lats=self.grid.lats, lons=lons)

        return recinfo, grid, data

    def load_profile(self, day, hour, coord, sfc=False):
        """
        return the full field (recinfo, self.grid, data) for a given level
        and variable

        Args:
            day (int): selected day
            hour (int): selected hour
            coord: coordinates (lat, lon)
            sfc (optional, bool): include surface data

        Returns:
            profile, sfcdata, indexinfo, (latindex, lonindex)
        """

        if coord[1] < 0:
            coord = (coord[0], coord[1]+360)
        logger.info(coord)
        latindex = argnearest(self.grid.lats, coord[0])
        lonindex = argnearest(self.grid.lons, coord[1])
        logger.info("latindex, lonindex {} {}".format(latindex, lonindex))
        logger.info(self.grid.lats[latindex])
        logger.info(self.grid.lons[lonindex])

        zeros = np.zeros((self.headerinfo['Nz']-1))
        if self.resolution == 1.0:
            profile = {
                'HGTS': zeros.copy(), 'TEMP': zeros.copy(),
                'UWND': zeros.copy(), 'VWND': zeros.copy(),
                'WWND': zeros.copy(), 'RELH': zeros.copy(),
                'PRSS': zeros.copy(),
                }
            press_variable = 'PRSS'
        elif self.resolution == 0.5:
            profile = {
                'HGTS': zeros.copy(), 'TEMP': zeros.copy(),
                'UWND': zeros.copy(), 'VWND': zeros.copy(),
                'SPHU': zeros.copy(), 'PRES': zeros.copy(),
                'SIGMA': zeros.copy()
                }
            press_variable = 'SIGMA'

            # get the sfc pressure
            bin_index = calc_index(
                self.indexinfo,
                self.headerinfo,
                self.levels,
                day,
                hour,
                0,
                'PRSS'
                )

            recinfo, data = read_data(
                self.fname,
                bin_index,
                self.headerinfo,
                stopat=(latindex + 1, lonindex + 1)
                )
            sfc_pres = interp(
                data[lonindex:lonindex+2, latindex:latindex+2],
                self.grid.lats[latindex:latindex+2],
                self.grid.lons[lonindex:lonindex+2],
                coord
                )
            bin_index = calc_index(
                self.indexinfo,
                self.headerinfo,
                self.levels,
                day,
                hour,
                0,
                'T02M'
                )
            recinfo, data = read_data(
                self.fname,
                bin_index,
                self.headerinfo,
                stopat=(latindex+1, lonindex+1)
                )
            sfc_temp = interp(
                data[lonindex:lonindex+2, latindex:latindex+2],
                self.grid.lats[latindex:latindex+2],
                self.grid.lons[lonindex:lonindex+2],
                coord
                )

        elif self.resolution == 0.25:
            profile = {
                'HGTS': zeros.copy(), 'TEMP': zeros.copy(),
                'UWND': zeros.copy(), 'VWND': zeros.copy(),
                'WWND': zeros.copy(), 'RELH': zeros.copy(),
                'PRES': zeros.copy(), 'SIGMA': zeros.copy()
                }
            press_variable = 'SIGMA'

            # get the sfc pressure
            bin_index = calc_index(
                self.indexinfo,
                self.headerinfo,
                self.levels,
                day,
                hour,
                0,
                'PRSS'
                )
            recinfo, data = read_data(
                self.fname,
                bin_index,
                self.headerinfo,
                stopat=(latindex+1, lonindex+1)
                )
            sfc_pres = interp(
                data[lonindex:lonindex+2, latindex:latindex+2],
                self.grid.lats[latindex:latindex+2],
                self.grid.lons[lonindex:lonindex+2],
                coord
                )
            bin_index = calc_index(
                self.indexinfo,
                self.headerinfo,
                self.levels,
                day,
                hour,
                0,
                'T02M'
                )
            recinfo, data = read_data(
                self.fname, bin_index,
                self.headerinfo,
                stopat=(latindex+1, lonindex+1)
                )
            sfc_temp = interp(
                data[lonindex:lonindex+2, latindex:latindex+2],
                self.grid.lats[latindex:latindex+2],
                self.grid.lons[lonindex:lonindex+2],
                coord
                )
        else:
            raise ValueError("unknown resolution")

        # read the indexinfo at this timestep
        rec_length = (self.headerinfo['Nx'] * self.headerinfo['Ny']) + 50
        bin_index = calc_index(
            self.indexinfo,
            self.headerinfo,
            self.levels,
            day,
            hour,
            0,
            'PRSS'
            ) - rec_length

        with open(self.fname, mode='rb') as f:
            f.seek(bin_index, 0)
            unpacked = b''.join(struct.unpack(50*'s', f.read(50)))
            indexinfo = convertindexlist(
                split_format(6 * 'i2,'+'s2,s4,i4,f14,f14', unpacked)
                )
        logger.info('indexinfo at timestep {}'.format(indexinfo))

        # skip the surface level
        for i in range(1, self.headerinfo['Nz']):
            logger.info(
                "reading level {} of {}".format(i, self.headerinfo['Nz'])
                )
            for variable in map(lambda x: x[0], self.levels[i]['vars']):
                level = self.levels[i]['level']
                bin_index = calc_index(
                    self.indexinfo,
                    self.headerinfo,
                    self.levels,
                    day,
                    hour,
                    level,
                    variable
                    )

                recinfo, data = read_data(
                    self.fname,
                    bin_index,
                    self.headerinfo,
                    stopat=(latindex+1, lonindex+1)
                    )
                val = data[lonindex, latindex]
                profile[variable][i-1] = val
                profile[press_variable][i-1] = level

        if self.resolution == 0.5:
            # calculate the heights from hypsometric formula
            logger.info(
                "p from sigma {} {}".format(
                    calc_p_from_sigma(profile["SIGMA"], sfc_pres)
                    )
                )
            temps = np.concatenate(([sfc_temp], profile['TEMP']))
            logger.info('temperatures {}'.format(temps))
            layer_mean_temp = temps[:-1] + (temps[1:] - temps[:-1]) / 2.
            logger.info('layer mean {}'.format(layer_mean_temp))
            pres = np.concatenate(([sfc_pres], profile['PRES']))
            Rd = 287.04
            g = 9.81
            deltah = Rd / g * layer_mean_temp * np.log(pres[:-1] / pres[1:])
            logger.info('delta h {}'.format(deltah))
            logger.info('heights {}'.format(np.cumsum(deltah)))
            profile['HGTS'] = np.cumsum(deltah)
            profile['PRSS'] = profile['PRES']
            profile['RELH'] = rh_from_q(
                profile['PRSS'],
                profile['SPHU'],
                profile['TEMP']
                )

            logger.info(profile)
            # calculate the relative humidity..

        if self.resolution == 0.25:
            # calculate the heights from hypsometric formula
            logger.info("p from sigma {} {}".format(
                calc_p_from_sigma(profile["SIGMA"], sfc_pres)
                ))
            temps = np.concatenate(([sfc_temp], profile['TEMP']))
            logger.info('temperatures {}'.format(temps))
            layer_mean_temp = temps[:-1] + (temps[1:] - temps[:-1]) / 2.
            logger.info('layer mean {}'.format(layer_mean_temp))
            pres = np.concatenate(([sfc_pres], profile['PRES']))
            Rd = 287.04
            g = 9.81
            deltah = Rd / g * layer_mean_temp * np.log(pres[:-1] / pres[1:])
            deltah = np.abs(deltah)
            logger.info('delta h {}'.format(deltah))
            logger.info('heights {}'.format(np.cumsum(deltah)))
            profile['HGTS'] = np.cumsum(deltah)
            profile['PRSS'] = profile['PRES']

            logger.info(profile)
            # calculate the relative humidity..

        sfcdata = {}
        if sfc:
            for sfcvar in map(lambda x: x[0], self.levels[0]['vars']):
                bin_index = calc_index(
                    self.indexinfo,
                    self.headerinfo,
                    self.levels,
                    day,
                    hour,
                    0,
                    sfcvar
                    )
                recinfo, data = read_data(
                    self.fname, bin_index,
                    self.headerinfo,
                    stopat=(latindex+1, lonindex+1)
                    )
                val = interp(
                    data[lonindex:lonindex+2, latindex:latindex+2],
                    self.grid.lats[latindex:latindex+2],
                    self.grid.lons[lonindex:lonindex+2],
                    coord
                    )
                sfcdata[sfcvar] = val
        return profile, sfcdata, indexinfo, (latindex, lonindex)


class Downloader():
    """
    Download the global GDAS1 file from ARL server.
    """

    def __init__(self, saveFolder, *args, bufferSize=10*1024, flagForce=False):
        """
        Downloader initialization.

        Parameters
        ----------
        saveFolder: str
        folder for saving the GDAS1 global dataset.

        Keywords
        --------
        bufferSize: integer
        buffer size for the downloading queue.
        flagForce: boolean
        flag to control whether to overwrite the GDAS1 gobal dataset.

        History
        -------
        2019-09-30. First edition by Zhenping
        """

        self.ftpHost = FTPHost
        self.ftpFolder = FTPFolder
        self.buffer = bufferSize
        self.dlSize = 0   # downloaded bytes counter
        self.flagForce = flagForce
        if os.path.exists(saveFolder):
            self.saveFolder = saveFolder
        else:
            os.mkdir(saveFolder)
            self.saveFolder = saveFolder

    def startConnection(self):
        """
        start the connection with the FTP server.
        """

        try:
            self.ftp = FTP(self.ftpHost)
            self.ftp.login()
            logger.info(self.ftp.getwelcome())
            # open the GDAS1 folder
            self.ftp.cwd(self.ftpFolder)
        except Exception as e:
            logger.error('Failure in FTP connection.')
            raise e

    def closeConnection(self):
        self.ftp.close()

    def checkFiles(self, dt):
        """
        check whether the GDAS1 file for the given time, has been saved in
        you local machine.
        """

        file = fname_from_date(dt)
        if os.path.exists(os.path.join(self.saveFolder, file)):
            return True
        else:
            return False

    def download(self, dt):
        """
        Download the GDAS1 file.
        """

        dlFile = fname_from_date(dt)
        self.dlSize = 0

        # check whether the file exists
        flag = False
        if self.checkFiles(dt) and not self.flagForce:
            logger.warn('{file} exists in {folder}.'.format(
                file=dlFile, folder=self.saveFolder))
            return
        else:
            # connect the server
            self.startConnection()

            for file in self.ftp.nlst():
                if file == dlFile:
                    flag = True

        lastDisSize = 0

        def dlCallback(data, dlFile, fileSize, f):
            """
            Display the download percentage.
            """

            nonlocal lastDisSize
            self.dlSize = self.dlSize + len(data)
            if (self.dlSize - lastDisSize) >= (fileSize * 0.005):
                logger.info('Download {file} {percentage: 04.1f}%'.format(
                    file=dlFile, percentage=(self.dlSize / fileSize) * 100))
                lastDisSize = self.dlSize
            f.write(data)

        if flag:
            self.ftp.sendcmd("TYPE i")    # Switch to Binary mode
            fileSize = self.ftp.size(dlFile)
            self.ftp.sendcmd("TYPE a")    # Swich back to Ascii mode

            with open(os.path.join(self.saveFolder, dlFile), 'wb') as fHandle:
                self.ftp.retrbinary(
                    'RETR {file}'.format(file=dlFile),
                    lambda block: dlCallback(
                                                block,
                                                dlFile,
                                                fileSize,
                                                fHandle
                                            ),
                    blocksize=self.buffer
                    )
        else:
            logger.warn(
                '{file} doesn\'t exist in the server.'.format(file=dlFile)
                )

        self.closeConnection()


class ArgumentParser(argparse.ArgumentParser):
    """
    Override the error message for argparse.

    reference
    ---------
    https://stackoverflow.com/questions/5943249/python-argparse-and-controlling-overriding-the-exit-status-code/5943381
    """

    def _get_action_from_name(self, name):
        """Given a name, get the Action instance registered with this parser.
        If only it were made available in the ArgumentError object. It is
        passed as it's first arg...
        """
        container = self._actions
        if name is None:
            return None
        for action in container:
            if '/'.join(action.option_strings) == name:
                return action
            elif action.metavar == name:
                return action
            elif action.dest == name:
                return action

    def error(self, message):
        exc = sys.exc_info()[1]
        if exc:
            exc.argument = self._get_action_from_name(exc.argument_name)
            raise exc
        super(ArgumentParser, self).error(message)


def extractorStation(year, month, day, hour, lat, lon, station, *args,
                     saveFolder='',
                     globalFolder='',
                     force=False):
    """
    extract the GDAS1 profile for a given coordination and time.
    """

    if (not os.path.exists(saveFolder)):
        logger.error('{} does not exist.'.format(saveFolder))
        logger.error('Please set the folder for saving the GDAS1 profiles.')
        raise ValueError

    if (not os.path.exists(globalFolder)):
        logger.error('{} does not exist.'.format(globalFolder))
        logger.error('Please specify the folder of the GDAS1 global files.')
        raise ValueError

    dt = datetime.datetime(year, month, day, hour)

    globalFile = fname_from_date(dt)
    # download GDAS1 global dataset
    logger.warn('Global file does not exist. Check {file}.'.
                format(file=globalFile))
    logger.info('Start to download {}'.format(globalFile))
    downloader = Downloader(globalFolder, flagForce=force)
    downloader.download(dt)

    # extract the profile
    profile, sfcdata, indexinfo, ind = reader(
            os.path.join(globalFolder, globalFile)
        ).load_profile(day, hour, (lat, lon))

    # write to ASCII file
    profileFile = '{station}_{lat:6.2f}_{lon:6.2f}_{date}_{hour}.gdas1'.format(
                                station=station,
                                lat=lat,
                                lon=lon,
                                date=dt.strftime('%Y%m%d'),
                                hour=dt.strftime('%H')
                                )
    write_profile(
                        os.path.join(saveFolder, profileFile),
                        indexinfo,
                        ind,
                        (lat, lon),
                        profile,
                        sfcdata
                    )

    logger.info('Finish writing profie {}'.format(profileFile))


def main():
    """
    Command line interface.
    """
    # Define the command line arguments.
    description = 'extract the GDAS1 profile from GDAS1 global binary data.'
    parser = ArgumentParser(
                                prog='ARLreader',
                                description=description,
                                formatter_class=RawTextHelpFormatter
                            )

    # Setup the arguments
    helpMsg = "start time of your interests (yyyymmdd-HHMMSS)." +\
              "\ne.g.20151010-120000"
    parser.add_argument("-s", "--start_time",
                        help=helpMsg,
                        dest='start_time',
                        default='20990101-120000')
    helpMsg = "stop time of your interests (yyyymmdd-HHMMSS)." +\
              "\ne.g.20151010-120000"
    parser.add_argument("-e", "--end_time",
                        help=helpMsg,
                        dest='end_time',
                        default='20990101-120000')
    helpMsg = "latitude of your station. (-90, 90).\n" +\
              "Default: 30.533721"
    parser.add_argument("--latitude",
                        help=helpMsg,
                        dest='latitude',
                        default=30.533721,
                        type=float)
    helpMsg = "longitude of your station. (0, 360).\n" +\
              "Default: 114.367216"
    parser.add_argument("--longitude",
                        help=helpMsg,
                        dest='longitude',
                        default=114.367216,
                        type=float)
    helpMsg = "station name.\n" +\
              "Default: wuhan"
    parser.add_argument("--station",
                        help=helpMsg,
                        dest='station',
                        default='wuhan',
                        type=str)
    helpMsg = "folder for saving the GDAS1 global files.\n" +\
              "e.g., 'C:\\Users\\zhenping\\Desktop\\global'"
    parser.add_argument("-f", "--global_folder",
                        help=helpMsg,
                        dest='global_folder',
                        default='')
    helpMsg = "folder for saving the extracted profiles\n" +\
              "e.g., 'C:\\Users\\zhenping\\Desktop\\wuhan'"
    parser.add_argument("-o", "--profile_folder",
                        help=helpMsg,
                        dest='profile_folder',
                        default='')
    helpMsg = "force to download the GDAS1 global dataset (not suggested)"
    parser.add_argument("--force",
                        help=helpMsg,
                        dest='force',
                        action='store_true')

    # if no input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        # error info can be obtained by using e.argument_name and e.message
        logger.error('Error in parsing the input arguments. Please check ' +
                     'your inputs.\n{message}'.format(message=e.message))
        raise ValueError

    # run the command
    # extract gdas1 profile
    start = datetime.datetime.strptime(args.start_time, '%Y%m%d-%H%M%S')
    end = datetime.datetime.strptime(args.end_time, '%Y%m%d-%H%M%S')
    hours = (end-start).days*24 + (end-start).seconds/3600
    timeList = [start + datetime.timedelta(hours=3*x)
                for x in range(0, int(hours/3))]   # temporal interval 3-hour
    saveFolder = args.profile_folder
    globalFolder = args.global_folder
    longitude = args.longitude
    latitude = args.latitude
    station_name = args.station

    for time in timeList:
        extractorStation(
                        time.year,
                        time.month,
                        time.day,
                        time.hour,
                        latitude,
                        longitude,
                        station_name,
                        saveFolder=saveFolder,
                        globalFolder=globalFolder,
                        force=args.force
                     )


if __name__ == '__main__':
    main()
