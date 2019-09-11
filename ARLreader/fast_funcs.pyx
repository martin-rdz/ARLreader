cimport cython
cimport numpy as np
import numpy as np

#
# ~/anaconda3/bin/python3 setup.py build_ext --inplace
#

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef unpack_data(long long[:] binarray, int nx, int ny, float initval, float exp, float prec, tuple stopat):
    """ unpack the binary data with the option to stop at a given index """
    cdef double[:,:] data = np.zeros((nx,ny))
    cdef float value_old = initval
    cdef float val

    cdef int nx_max = nx
    cdef int ny_max = ny
    cdef float scaled_exp = (2**float(7-exp))

    if stopat != (-1, -1):
        nx_min = max(stopat[1]-1,0)
        nx_max = min(stopat[1]+1,nx)
        ny_min = max(stopat[0]-1,0)
        ny_max = min(stopat[0]+1,ny)
    else:
        nx_min = 0
        nx_max = nx
        ny_min = 0
        ny_max = ny
    #print("nx ", nx_min, nx_max, " ny ", ny_min, ny_max)
    
    cdef int j
    cdef int ri
    for j in range(ny_max):
        ri = j*nx
        data[0,j] = ((binarray[ri]-127)/scaled_exp) + value_old
        value_old = data[0,j]


    for j in range(ny_min, ny_max):
        value_old = data[0,j]
        for i in range(1,nx_max):
            ri = j*nx + i
            #print(j,i, '->', ri)
            val = (binarray[ri]-127)/scaled_exp + value_old
            value_old = val
            if abs(val) < prec:
                #print(abs(val), '<', prec)
                val = 0.
            data[i,j] = val

    return np.asarray(data)