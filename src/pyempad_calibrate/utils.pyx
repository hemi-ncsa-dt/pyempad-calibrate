import numpy as np
cimport numpy as cnp
cnp.import_array()
cimport cython

VAL_DTYPE = np.uint32
DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.uint32_t VAL_DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def combine_direct(
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] g1,
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] g2,
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] off,
    str filename,
    int nsize,
):
    """
    Combine the data from the file using the given parameters.

    Conserves memory by reading the file in chunks and processing each chunk.
    Slighly slower than the `combine` function.

    Parameters
    ----------
    g1 : np.ndarray (128*128*2,)
        The low gain callibration data. Even and odd values are stored in the same array.

    g2 : np.ndarray (128*128*2,)
        The digital gain calibration data. Even and odd values are stored in the same array.

    off : np.ndarray (128*128*2,)
        The offset array for low gain calibration data. Even and odd values are stored
        in the same array.

    filename : str
        The filename of the data file.

    nsize : int
        The number of frames in the file.

    Returns
    -------
    np.ndarray
        The combined frames.
    """

    cdef cnp.ndarray[DTYPE_t, ndim=1] frames = np.zeros(128*128*nsize, dtype=DTYPE)
    cdef int i, j, k
    cdef DTYPE_t ana, dig, gn
    cdef cnp.uint32_t ana_mask = 0x3FFF
    cdef cnp.uint32_t dig_mask = 0x3FFFC000
    cdef cnp.uint32_t gn_mask = 0x80000000
    cdef int wrap = 128 * 128 * 2
    cdef cnp.ndarray[VAL_DTYPE_t, ndim=1] values

    with open(filename, 'rb') as f:
        i = 0
        while True:
            chunk = f.read(128*128*128*4)
            if not chunk:
                break
            values = np.frombuffer(chunk, dtype=VAL_DTYPE)
            for k in range(values.shape[0]):
                j = i % wrap
                ana = values[k] & ana_mask
                dig = (values[k] & dig_mask) >> 14
                gn  = (values[k] & gn_mask) >> 31

                frames[i] = ana * (1.0 - gn) + g1[j] * (ana - off[j]) * gn + g2[j] * dig
                i += 1
    return frames

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def combine(
    cnp.ndarray[VAL_DTYPE_t, ndim=1, mode="fortran"] values,
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] g1,
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] g2,
    cnp.ndarray[DTYPE_t, ndim=1, mode="fortran"] off
):
    cdef cnp.ndarray[DTYPE_t, ndim=1] frames = np.zeros_like(values, dtype=DTYPE)
    cdef int i, j
    cdef DTYPE_t ana, dig, gn=0.01
    cdef cnp.uint32_t ana_mask = 0x3FFF
    cdef cnp.uint32_t dig_mask = 0x3FFFC000
    cdef cnp.uint32_t gn_mask = 0x80000000
    cdef int wrap = 128 * 128 * 2
    for i in range(values.shape[0]):
        j = i % wrap
        ana = values[i] & ana_mask
        dig = (values[i] & dig_mask) >> 14
        gn  = (values[i] & gn_mask) >> 31

        frames[i] = ana * (1.0 - gn) + g1[j] * (ana - off[j]) * gn + g2[j] * dig
    return frames

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def debounce(
    cnp.ndarray[DTYPE_t, ndim=3, mode="fortran"] frames,
    int wide,
    int w2
):
    cdef int j, w, histMaxArg
    cdef Py_ssize_t i
    cdef DTYPE_t nNumPoint = 2 * w2 + 1
    cdef DTYPE_t histMaxVal

    cdef DTYPE_t lbound = -200 - (wide / 2)
    cdef DTYPE_t ubound = 220 - (wide / 2)
    cdef cnp.ndarray[DTYPE_t, ndim=1] nEdges = np.arange(lbound, ubound + wide, wide, dtype=DTYPE)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] histVal = np.zeros(nEdges.shape[0] - 1, dtype=np.int64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] histEdge = np.zeros(nEdges.shape[0], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] wVal = np.arange(-w2, w2 + 1, 1, dtype=DTYPE)
    cdef int nInd1, nInd2
    cdef DTYPE_t sum_y, sum_xy, sum_x2y, sum_x2, sum_x4, comx, aVal, bVal, offset
    cdef Py_ssize_t num_frames = frames.shape[2]

    sum_y = sum_xy = sum_x2y = sum_x2 = sum_x4 = comx = 0
    for i in range(num_frames):
        histVal, histEdge = np.histogram(frames[:,:,i], bins=nEdges)
        histMaxArg = np.argmax(histVal)
        histMaxVal = histVal[histMaxArg]
        if histMaxVal <= 40:
            continue

        nInd1 = max(0, histMaxArg - w2)
        nInd2 = min(histMaxArg + w2 + 1, histVal.shape[0])
        sum_y = sum_xy = sum_x2y = sum_x2 = sum_x4 = comx = 0
        w = 0
        for j in range(nInd1, nInd2):
            sum_y += histVal[j]
            sum_xy += wVal[w] * histVal[j]
            sum_x2y += wVal[w] * wVal[w] * histVal[j]
            sum_x2 += wVal[w] * wVal[w]
            sum_x4 += wVal[w] * wVal[w] * wVal[w] * wVal[w]
            w += 1

        bVal = sum_xy / sum_x2
        aVal = (nNumPoint * sum_x2y - sum_x2 * sum_y) / (nNumPoint * sum_x4 - sum_x2 * sum_x2)
        if abs(aVal) > 0.0001:
            comx = -bVal / (2 * aVal)
        offset = histEdge[histMaxArg] + (wide / 2) + comx * wide
        if abs(offset) > 200:
            continue
        frames[:,:,i] = frames[:, :, i] - offset
