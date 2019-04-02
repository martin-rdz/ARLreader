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


gridtup = collections.namedtuple('gridtup', 'lats, lons')

def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis
    Args:
        array (np.array): sorted array with values
        value: value to find
    Returns:
        index  
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
    val = np.log(e/ 6.112)
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
    th_e = th_l * np.exp((3.036 / t_l - 0.00178) * mr * (1+ 0.448e-3 * mr))
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
    assert all([isinstance(u,np.ndarray), isinstance(v,np.ndarray)]), \
        'input needs to be numpy array' 
    wdir = 90. - (np.arctan2(-v, -u)*180./np.pi)
    wdir[wdir < 0] += 360.
    wvel = np.sqrt(u * u + v * v)
    return wdir, wvel    
	
	
def fname_from_date(dt):
    '''
    Args:
        dt (datetime): date

    Returns:
        name for the gdas file, eg "gdas1.oct14.w1"
    '''
    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
              7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    week_no = ((dt.day-1)//7)+1
    return 'gdas1.{}{}.w{}'.format(months[dt.month], dt.strftime('%y'), week_no)
	
		
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
        #print('fmt code ', fmt)
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
        'y m d h fc lvl grid name exp prec initval')
    return recordinfo(y=l[0], m=l[1], d=l[2], h=l[3],
                      fc=l[4], lvl=l[5], grid=l[6], name=l[7],
                      exp=l[8], prec=l[9], initval=l[10])


def calc_no_within_ts(levels, lvl, varname):
    """calculate the position for a given level and variable inside the current timestep

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
            #print('level reached ', lvl)
            steps += list(map(lambda x: x[0], levels[i]['vars'])).index(varname)
            break
        steps += len(levels[i]['vars'])
    return steps


def get_lvl_index(levels, lvl):
    """index of a given level in levels"""
    for k, v in levels.items():
        if v['level'] == lvl:
            return k, v


def calc_index(indexinfo, headerinfo, levels, day, hour, lvl, varname):
    """calculate the starting byte for a given time, level and variable

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
    rec_length = (headerinfo['Nx']*headerinfo['Ny'])+50
    no_vars = sum([len(v['vars']) for v in levels.values()])
    # important for each timestep you have to jup a whole record (index + header + variables (+empty) are replicated)
    timesteplength = (rec_length * no_vars) + rec_length
    #print('timesteplength', timesteplength)
    no_full_slices = ((day - indexinfo.d)*8 + int(hour/3)) * timesteplength
    no_at_timestep = calc_no_within_ts(levels, lvl, varname) * rec_length
    #print('no_full_slices', no_full_slices, 'no_at_timestep', no_at_timestep)
    # plus the initial record length
    assert no_full_slices + no_at_timestep + rec_length > 0, 'calculated negative index, check your inputs'
    return no_full_slices + no_at_timestep + rec_length


def read_data(fname, bin_index, headerinfo, stopat=None):
    """read the data from a given file based on the starting index and decode the data with unpack_data()

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
        recinfo = convertindexlist(split_format(7*'i2,'+'s4,i4,f14,f14', unpacked))
        unpacked = struct.unpack((rec_length-50)*'B', f.read(rec_length-50))
        data = unpack_data(np.array(unpacked), headerinfo['Nx'], headerinfo['Ny'], 
               recinfo.initval, recinfo.exp, recinfo.prec, stopat=stopat)
    return recinfo, data

def unpack_data(binarray, nx, ny, initval, exp, prec, stopat=None):
    """ unpack the binary data with the option to stop at a given index """
    data = np.zeros((nx,ny))
    value_old = initval
    
    nx_max = nx
    ny_max = ny
    if stopat != None:
        nx_max = stopat[0]
        ny_max = stopat[1]
   
    for j in range(ny):
        ri = j*nx
        data[0,j] = (binarray[ri]-127)/(2**(7-exp)) + value_old
        value_old = data[0,j]

    for j in range(ny):
        value_old = data[0,j]
        for i in range(1,nx):
            ri = j*nx + i
            #print(j,i, '->', ri)
            val = (binarray[ri]-127)/(2**(7-exp)) + value_old
            value_old = val
            if abs(val) < prec:
                #print(abs(val), '<', prec)
                val = 0.
            data[i,j] = val
            if stopat != None:
                if i >= stopat[0] and j >= stopat[1]:
                    break
    return data


def calc_p_from_sigma(sigma, p_sfc):
    """calculate a levels pressure from sigma coordinate and ground pressure

    Example from https://ready.arl.noaa.gov/data/archives/gdas0p5/readme_gdas0p5_info.txt:

    The pressure level is defined as: P = A + .B*PRSS,
    where PRSS is the surface pressure and A.B is defined as the level 
    (ie., 16.853 = 869 hPa when the surface pressure is 1000 hPa
    [P=16+0.853*1000])
    """
    A = np.floor(sigma)
    B = sigma - A
    print("A", A)
    print("B", B)
    print('pressure ', A+(B*p_sfc))
    return  A+(B*p_sfc)


def interp(data, lats, lons, coord):
    #print('interp ', data, coord, lats, lons)
    return interp2d(data, lons, lats, (coord[1], coord[0]))
    
    
def interpx(arr, grid, coord):
    return np.transpose((coord-grid[0])*(arr[:,1]-arr[:,0])/(grid[1]-grid[0])+arr[:,0])


def interp2d(arr, grid0, grid1, coord):
    """2d interpolation of the values between the grid cells"""
    val = interpx(arr, grid0, coord[0])
    return (coord[1]-grid1[0])*(val[1]-val[0])/(grid1[1]-grid1[0])+val[0]


def write_profile(fname, headerinfo, ind, coord, profile, sfcdata):
    """write the profile to a file

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
        f.write(' Profile Time:  {:2d} {:2d} {:2d} {:2d}  0\n'.format(headerinfo.y, headerinfo.m, headerinfo.d, headerinfo.h))
        f.write(' Used Nearest Grid Point ( {:3d}, {:3d}) to Lat:   {:5.2f}, Lon:    {:5.2f}\n'.format(ind[1], ind[0], coord[0], coord[1]))
        f.write('        2D Fields \n')
        f.write('\n\n\n')
        f.write('        3D Fields \n')
        f.write('        HGTS  TEMP  UWND  VWND  WWND  RELH     TPOT  WDIR  WSPD\n')
        f.write('           m    oC   m/s   m/s  mb/h     %       oK   deg   m/s\n')
        for i in range(23):
            f.write('  {:4.0f} {:5.0f} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}    {:5.1f} {:5.1f}  {:4.1f}\n'.format(
                profile['PRSS'][i], profile['HGTS'][i], profile['TEMP'][i]-273.15, profile['UWND'][i], profile['VWND'][i],
                profile['WWND'][i]*3600, profile['RELH'][i], potT[i], wdir[i], wvel[i]))


class reader():
    """read the header of the gdas file
    50 index bytes, 108 grid bytes and the vars at different heights

    Args:
        fname: filename
    """
    def __init__(self, fname):
        self.fname = fname
        with open(fname,mode='rb') as f:
            content = f.read(5000)
            # for python struct s,c,p are chars
            unpacked = b''.join(struct.unpack(50*'s',content[:50]))
            index_info = split_format(7*'i2,'+'s4,i4,f14,f14', unpacked)
            indexinfo = convertindexlist(index_info)
            print('indexinfo ', indexinfo)
            self.indexinfo = indexinfo
            unpacked = b''.join(struct.unpack_from(108*"s",content[:], 50))
            header_format = 's4,i3,i2,'+12*'f7,'+3*'i3,'+'i2,i4'
            header = split_format(header_format, unpacked)
            headerinfo = {'source': header[0], 'fcth': header[1],
                          'minDatatime': header[2], 'griddef': header[3:15],
                          'Nx': header[15], 'Ny': header[16], 'Nz': header[17],
                          'Coordzflag': header[18], 'headerlength': header[19]}
            print('raw header ', header)
            print('headerinfo ', headerinfo)
            assert headerinfo['source'] in ['GDAS', 'GFSG'], 'other sources than gdas not supported yet'
            self.headerinfo = headerinfo
            if headerinfo['Nx'] == 720 and headerinfo['Ny'] == 361:
                self.resolution = 0.5
            elif headerinfo['Nx'] == 360 and headerinfo['Ny'] == 181:
                self.resolution = 1.0
            else:
                raise ValueError("could not infer GDAS resolution")
            # read the variables in the different levels
            levels = {}
            cur = 158
            for ih in range(headerinfo['Nz']):
                # unpack the varinfos
                unpacked = b''.join(struct.unpack_from(8*"s",content[:], cur))
                height_lvl, Nvars = split_format('f6,i2', unpacked)
                #print(ih, height_lvl, Nvars, unpacked)
                cur += 8
                # print('at ', height_lvl, ' no vars ', Nvars)
                levels[ih] = {'level': height_lvl, 'vars': []}
                # go through all vars at this height level
                for ivar in range(Nvars):
                    unpacked = b''.join(struct.unpack_from(8*"s",content[:], cur))
                    cur += 8
                    varname, checksum = split_format('s4,i3,x1', unpacked)
                    #print(ivar, varname, checksum)
                    levels[ih]['vars'].append((varname, checksum))
            # print(levels)
            self.levels = levels
            
            self.grid = gridtup(lats=np.linspace(-90, 90, headerinfo['Ny']), 
                                lons=np.linspace(0, 359, headerinfo['Nx']))

            
    def load_heightlevel(self, day, hour, level, variable, truelon=True):
        """return the full field (recinfo, self.grid, data) for a given level and variable

        Args:
            day (int): selected day
            hour (int): selected hour
            level: selected level
            variable: selected variable
            truelon (optional, bool): convert to true  lon
            
        Returns:
            recordinfo, grid, data
        """
        assert hour%3 == 0, 'Other time resolution than 3h not supported yet.'
        print(self.levels, level)
        varlist_at_level = list(map(lambda x: x[0], get_lvl_index(self.levels, level)[1]['vars']))
        assert variable in varlist_at_level, 'Variable not available at this level. Only: ' + str(varlist_at_level)
        bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, level, variable)
        recinfo, data = read_data(self.fname, bin_index, self.headerinfo)
        print('recordinfo ', recinfo)
        assert all([recinfo.name == variable, recinfo.d == day, recinfo.h == hour]), \
            'Something went wrong while reading file'

        grid = self.grid
        if truelon:
            print('convert to the true longitude coordinates')
            data = np.roll(data, -181, axis=0)
            lons = np.roll(self.grid.lons, -181, axis=0)
            lons[lons > 180] -= 360.
            grid = gridtup(lats=self.grid.lats, lons=lons)
        return recinfo, grid, data

    
    def load_profile(self, day, hour, coord, sfc=False):
        """return the full field (recinfo, self.grid, data) for a given level and variable

        Args:
            day (int): selected day
            hour (int): selected hour
            coord: coordinates (lat, lon)
            sfc (optional, bool): include surface data
            
        Returns:
            profile, sfcdata, indexinfo, (latindex, lonindex)
        """
        # HGTS TEMP UWND VWND WWND RELH
        #latindex = np.where(self.grid.lats == min(self.grid.lats, key= lambda t: abs(coord[0] - t)))[0].tolist()[0]
        #lonindex = np.where(self.grid.lons == min(self.grid.lons, key= lambda t: abs(coord[1] - t)))[0].tolist()[0]
        
        #coord = (coord[0], 180+coord[1])
        #assert np.all(np.diff(self.grid.lats) == 1)
        if coord[1] < 0:
            coord = (coord[0], coord[1]+360)
        print(coord)
        latindex = argnearest(self.grid.lats, coord[0])
        lonindex = argnearest(self.grid.lons, coord[1])
        print("latindex, lonindex ", latindex, lonindex)
        print(self.grid.lats[latindex])
        print(self.grid.lons[lonindex])
        
        zeros = np.zeros((self.headerinfo['Nz']-1))
        if self.resolution == 1.0:
            profile = {'HGTS': zeros.copy(), 'TEMP': zeros.copy(), 'UWND': zeros.copy(), 
                        'VWND': zeros.copy(), 'WWND': zeros.copy(), 'RELH': zeros.copy(),
                        'PRSS': zeros.copy(),}
            press_variable = 'PRSS'
        elif self.resolution == 0.5:
            profile = {'HGTS': zeros.copy(), 'TEMP': zeros.copy(), 'UWND': zeros.copy(), 
                       'VWND': zeros.copy(), 'SPHU': zeros.copy(),
                       'PRES': zeros.copy(), 'SIGMA': zeros.copy()}
            press_variable = 'SIGMA'

            # get the sfc pressure
            bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, 0, 'PRSS')
            recinfo, data = read_data(self.fname, bin_index, self.headerinfo, stopat=(latindex+1, lonindex+1))
            sfc_pres = interp(data[lonindex:lonindex+2, latindex:latindex+2],
                         self.grid.lats[latindex:latindex+2], self.grid.lons[lonindex:lonindex+2], coord)
            bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, 0, 'T02M')
            recinfo, data = read_data(self.fname, bin_index, self.headerinfo, stopat=(latindex+1, lonindex+1))
            sfc_temp = interp(data[lonindex:lonindex+2, latindex:latindex+2],
                         self.grid.lats[latindex:latindex+2], self.grid.lons[lonindex:lonindex+2], coord)

        # read the indexinfo at this timestep
        rec_length = (self.headerinfo['Nx']*self.headerinfo['Ny'])+50
        bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, 0, 'PRSS') - rec_length
        with open(self.fname, mode='rb') as f:
            f.seek(bin_index, 0)
            unpacked = b''.join(struct.unpack(50*'s', f.read(50)))
            indexinfo = convertindexlist(split_format(7*'i2,'+'s4,i4,f14,f14', unpacked))
        print('indexinfo at timestep', indexinfo)
        
        # skip the surface level
        for i in range(1, self.headerinfo['Nz']):
            for variable in map(lambda x: x[0], self.levels[i]['vars']):
            #for variable in ['HGTS']:
                level = self.levels[i]['level']
                bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, level, variable)
                recinfo, data = read_data(self.fname, bin_index, self.headerinfo, stopat=(latindex+1, lonindex+1))
                val = interp(data[lonindex:lonindex+2, latindex:latindex+2],
                             self.grid.lats[latindex:latindex+2], self.grid.lons[lonindex:lonindex+2], coord)
                #val = data[lonindex, latindex]
                #print(variable, level, data.shape, latindex, lonindex, data[lonindex, latindex])
                #print('value nearest ', data[lonindex, latindex], ' interp ', val)
                profile[variable][i-1] = val
                profile[press_variable][i-1] = level


        if self.resolution == 0.5:
            # calculate the heights from hypsometric formula
            print("p from sigma ", calc_p_from_sigma(profile["SIGMA"], sfc_pres))
            temps = np.concatenate(([sfc_temp], profile['TEMP']))
            print('temperatures', temps)
            layer_mean_temp = temps[:-1]+(temps[1:]-temps[:-1])/2.
            print('layer mean', layer_mean_temp)
            pres = np.concatenate(([sfc_pres], profile['PRES']))
            Rd = 287.04
            g = 9.81
            deltah = Rd/g*layer_mean_temp*np.log(pres[:-1]/pres[1:])
            print('delta h', deltah)
            print('heights', np.cumsum(deltah))
            profile['HGTS'] = np.cumsum(deltah)
            profile['PRSS'] = profile['PRES']
            profile['RELH'] = rh_from_q(profile['PRSS'], profile['SPHU'], profile['TEMP'])

            print(profile)
            # calculate the relative humidity..
        
        sfcdata = {}
        if sfc:
            for sfcvar in map(lambda x: x[0], self.levels[0]['vars']):
                bin_index = calc_index(self.indexinfo, self.headerinfo, self.levels, day, hour, 0, sfcvar)
                recinfo, data = read_data(self.fname, bin_index, self.headerinfo, stopat=(latindex+1, lonindex+1))
                val = interp(data[lonindex:lonindex+2, latindex:latindex+2],
                             self.grid.lats[latindex:latindex+2], self.grid.lons[lonindex:lonindex+2], coord)
                sfcdata[sfcvar] = val
        return profile, sfcdata, indexinfo, (latindex, lonindex)
