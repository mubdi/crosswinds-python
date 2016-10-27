#
# Function set for calculation of datacube correlation functions
# Coder: Mubdi Rahman
# First Created: 2016-10-04
# Last Modified: 2016-10-25
#

# BASIC OUTLINE
#  Given: Data Cube with dcube(x,y,vr) and vectors (c_x), (c_y), (c_vr)
#
## cython: profile=True

#### IMPORTS ####

import numpy as n
import datetime
cimport numpy as n 
cimport cython
from cython.parallel cimport prange, parallel
cimport openmp

# Needed for KD-Tree implementation
from scipy import spatial

#### DEFINED VARIABLES ####

DTYPE = n.double

ctypedef n.double_t DTYPE_t

cdef extern from 'cross_corr_2d_1d.c':
    double *cross_corr_2d_1d( double *norm_dcube, 
    int n_x, int n_y, int n_vr,
    double *c_x, double *c_y, 
    double *c_vr,
    double c_xi, double c_yi, double c_vri,
    double rmin, double rmax, double vrmin, double vrmax 
    ) nogil 

openmp.omp_set_dynamic(1)

#### TESTING FUNCTIONS ####

def gen_debug_dcube(xlims=(0, 1), ylims=(0, 1),
                    vellims=(-10, 10), dsize=(100, 100, 100), n_cent=30):
    """
    Generating a cube of random, gaussian distributed data centred at n_cent,
    along with the required axis coordinate lists
    """

    c_x = n.linspace(*xlims, num=dsize[0])
    c_y = n.linspace(*ylims, num=dsize[1])
    c_vr = n.linspace(*vellims, num=dsize[2])

    dcube = n.random.randn(*dsize) + n_cent

    return dcube, c_x, c_y, c_vr


def gen_debug_corr_dcube(xlims=(0, 1), ylims=(0, 1),
                         vellims=(-10, 10), dsize=(100, 100, 100),
                         n_cent=30, n_over=2, sig_over=0.5):
    """
    Generating a cube of random, gaussian distributed data centred at n_cent,
    as well as a gaussian overdensity at the centre of the box with n_over
    overdensity along with the required axis coordinate lists
    In this particular instantiation, the gaussian is spherically symmetric in
    both spatial and velocity axes.
    """

    c_x = n.linspace(*xlims, num=dsize[0])
    c_y = n.linspace(*ylims, num=dsize[1])
    c_vr = n.linspace(*vellims, num=dsize[2])

    dcube = n.random.randn(*dsize) + n_cent

    cent_x, cent_y, cent_vr = n.mean(xlims), n.mean(ylims), n.mean(vellims)
    mesh_x, mesh_y, mesh_vr = n.meshgrid(
        c_x - cent_x, c_y - cent_y, c_vr - cent_vr)

    dist = n.sqrt(mesh_x**2 + mesh_y**2 + mesh_vr**2)

    dcube += gaussian_function(dist, [n_over * n_cent, 0, sig_over])

    return dcube, c_x, c_y, c_vr


#### FUNCTIONS ####


def vel_corr_1d(dcube, c_vr, binpts, mass=True, norm=False,
                baseline=0.0):

    n_bin = n.size(binpts) - 1
    n_x = dcube.shape[0]
    n_y = dcube.shape[1]
    n_vr = dcube.shape[2]

    # Mean density per spaxel
    mean_dens = n.mean(dcube - baseline)

    # Normalizing the Data Cube along each velocity axis
    if norm:
        norm_dcube = (dcube - mean_dens) / mean_dens
    else:
        norm_dcube = (dcube - baseline) / mean_dens

    # Where the Cross-correlation Sum will go:
    corr_sum_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the Cross-correlation Mean will go:
    corr_mean_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the Standard Deviation will go:
    corr_std_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the number of cells will go:
    corr_cells = n.zeros((n_x, n_y, n_vr, n_bin))

    # Distance Vector to search on:
    dist_vect = make_dist_1d(c_vr, c_vr)

    # The Actual Cross-Correlation:

    # For each cross-correlation bin
    for i_bin in range(n_bin):
        vrmin = binpts[i_bin]
        vrmax = binpts[i_bin + 1]
        # For each velocity element
        for i_vr in range(n_vr):
            tmp_vr_ind = n.where(between_1d(dist_vect[i_vr, :], vrmin, vrmax))
            corr_sum_cube[:, :, i_vr, i_bin] = norm_dcube[
                :, :, tmp_vr_ind[0]].sum(axis=(2))
            corr_mean_cube[:, :, i_vr, i_bin] = norm_dcube[
                :, :, tmp_vr_ind[0]].mean(axis=(2))
            corr_std_cube[:, :, i_vr, i_bin] = norm_dcube[
                :, :, tmp_vr_ind[0]].std(axis=(2))
            corr_cells[:, :, i_vr, i_bin] = tmp_vr_ind[0].size

    # Number of point per cross-correlation element:
    n_pts_bin = n.sum(corr_cells, axis=(0, 1, 2))

    # Number of sampling points:
    n_samp_pts = n.product(corr_mean_cube.shape[:-1])

    # Determing weighting:
    if mass:

        # Filtering Negative Numbers in weighting:

        tmp_dcube = n.copy(dcube - baseline)
        tmp_dcube[n.where(tmp_dcube < 0)] = 0.0

        weight = tmp_dcube.reshape(
            (n_x, n_y, n_vr, 1)) / n.sum(tmp_dcube)  # Sum to 1
        final_corr = n.sum(corr_mean_cube * weight, axis=(0, 1, 2))
        final_err = n.sqrt(n.sum((corr_std_cube * weight)**2,
                                 axis=(0, 1, 2)))


    else:
        # For volume weighting, just use the normal mean
        weight = corr_cells.reshape(
            (n_x, n_y, n_vr, n_bin)) / n.sum(corr_cells)
        final_corr = n.sum(corr_mean_cube * weight, axis=(0, 1, 2))

        # Standard Error of Mean:
        final_err = n.sum((corr_std_cube)**2,
                          axis=(0, 1, 2))**0.5 / n_samp_pts

    return final_corr, final_err


def space_corr_2d(dcube, c_x, c_y, binpts, mass=True, norm=False,
                  baseline=0.0, debug=False):

    n_bin = n.size(binpts) - 1
    n_x = dcube.shape[0]
    n_y = dcube.shape[1]
    n_vr = dcube.shape[2]

    # Mean density per spaxel
    mean_dens = n.mean(dcube - baseline)

    if norm:
        norm_dcube = (dcube - mean_dens) / mean_dens
    else:
        norm_dcube = (dcube - baseline) / mean_dens

    # Where the Cross-correlation Sum will go:
    corr_sum_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the Cross-correlation Mean will go:
    corr_mean_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the number of cells will go:
    corr_cells = n.zeros((n_x, n_y, n_vr, n_bin))

    # Where the Standard Deviation will go:
    corr_std_cube = n.zeros((n_x, n_y, n_vr, n_bin))

    # The Actual Cross-Correlation:

    # For each cross-correlation bin
    for i_bin in range(n_bin):
        rmin = binpts[i_bin]
        rmax = binpts[i_bin + 1]
        # For each radial element
        for i_x in range(n_x):
            if debug:
                print("X-cells: %i" % i_x)
            for i_y in range(n_y):
                dist_vect = make_dist_2d([[c_x[i_x], c_y[i_y]]], c_x, c_y)
                tmp_r_ind = n.where(between_2d(dist_vect, rmin, rmax))

                corr_sum_cube[i_x, i_y, :, i_bin] = norm_dcube[
                    tmp_r_ind[1], tmp_r_ind[2], :].sum(axis=(0, 1))
                corr_mean_cube[i_x, i_y, :, i_bin] = norm_dcube[
                    tmp_r_ind[1], tmp_r_ind[2], :].mean(axis=(0, 1))
                corr_std_cube[i_x, i_y, :, i_bin] = norm_dcube[
                    tmp_r_ind[1], tmp_r_ind[2], :].std(axis=(0, 1))
                corr_cells[i_x, i_y, :, i_bin] = tmp_r_ind[0].size

    # Number of point per cross-correlation element:
    n_pts_bin = n.sum(corr_cells, axis=(0, 1, 2))

    # Number of sampling points:
    n_samp_pts = n.product(corr_mean_cube.shape[:-1])

    # Determing weighting:
    if mass:

        # Filtering Negative Numbers in weighting:

        tmp_dcube = n.copy(dcube - baseline)
        tmp_dcube[n.where(tmp_dcube < 0)] = 0.0

        weight = tmp_dcube.reshape(
            (n_x, n_y, n_vr, 1)) / n.sum(tmp_dcube)  # Sum to 1
        final_corr = n.sum(corr_mean_cube * weight, axis=(0, 1, 2))

        final_err = n.sqrt(n.sum((corr_std_cube * weight)**2,
                                 axis=(0, 1, 2)))

    else:
        # For volume weighting, just use the normal mean

        weight = corr_cells.reshape(
            (n_x, n_y, n_vr, n_bin)) / n.sum(corr_cells)
        final_corr = n.sum(corr_mean_cube * weight, axis=(0, 1, 2))

        # Standard Error of Mean:
        final_err = n.sum((corr_std_cube)**2,
                          axis=(0, 1, 2))**0.5 / n_samp_pts

    return final_corr, final_err

def vel_space_corr(n.ndarray[double, ndim=3] dcube, 
                   n.ndarray[double, ndim=1] c_x,
                   n.ndarray[double, ndim=1] c_y, 
                   n.ndarray[double, ndim=1] c_vr, 
                   n.ndarray[double, ndim=1] binpts_r, 
                   n.ndarray[double, ndim=1] binpts_v,
                   bint norm=False, 
                   double baseline=0.0,
                   bint debug=False,
                   str mode='bf'):
    """
    Convenience Function for various algorithms for 2D+1D
    Position + Velocity Cross-correlation  
    """

    if mode == 'bf':
        retval = vel_space_corr_bf(dcube, c_x, c_y, c_vr,
                                   binpts_r, binpts_v,
                                   norm=norm, baseline=baseline,
                                   debug=debug)

    else:
        print("Your mode \"%s\" was not recognized." % mode)

    return retval

@cython.wraparound(False)
@cython.boundscheck(False)
def vel_space_corr_kd(n.ndarray[double, ndim=3] dcube, 
                      n.ndarray[double, ndim=1] c_x, 
                      n.ndarray[double, ndim=1] c_y, 
                      n.ndarray[double, ndim=1] c_vr, 
                      n.ndarray[double, ndim=1] binpts_r, 
                      n.ndarray[double, ndim=1] binpts_v,
                      bint norm=False, 
                      double baseline=0.0,
                      bint debug=False):

    cdef int n_bin_r = n.size(binpts_r) - 1
    cdef int n_bin_v = n.size(binpts_v) - 1
    cdef int n_x = dcube.shape[0]
    cdef int n_y = dcube.shape[1]
    cdef int n_vr = dcube.shape[2]
    cdef object starttime = datetime.datetime.now()

    # Determining Mean Density per Spaxel
    cdef double mean_dens = n.mean(dcube - baseline)

    cdef n.ndarray[double, ndim=3] norm_dcube, weight

    if norm:
        norm_dcube = (dcube - mean_dens) / mean_dens
    else:
        norm_dcube = (dcube - baseline) / mean_dens

    # Where the Cross-correlation Mean will go:
    cdef n.ndarray[double, ndim=2] corr_mean_cube = n.zeros(
        (n_bin_r, n_bin_v))

    # Where the number of cells will go:
    cdef n.ndarray[double, ndim=2] corr_cells = n.zeros(
        (n_bin_r, n_bin_v))

    # Where the Standard Deviation will go:
    cdef n.ndarray[double, ndim=2] corr_std_cube = n.zeros(
        (n_bin_r, n_bin_v))

    # Determing weighting:
    # Filtering Negative Numbers in weighting:

    weight = n.copy(dcube - baseline)
    weight[n.where(weight < 0)] = 0.0

    weight = weight / n.sum(weight)  # Sum to 1

    # Stats on OpenMP threads
    cdef int num_threads, num_proc
    num_proc = openmp.omp_get_num_procs()

    print("Number of Procs available: %i" % num_proc)
    print("Timing: Cross-correlation Started at %s" % str(starttime))

    cdef int i_x, i_y, i_vr, i_bin_r, i_bin_v
    cdef double rmin, rmax, vrmin, vrmax
    cdef double c_xi, c_yi, c_vri
    cdef double sumx, sumx2, ncell, meanval, stdval, tmpweight
    cdef set tmp_rad_inds_u, tmp_rad_inds_d, tmp_vel_inds_u, tmp_vel_inds_d
    cdef n.ndarray[int, ndim=2] tmp_rad_inds
    cdef n.ndarray[double, ndim=2] tmp_data_slice, g_x, g_y, s_pts_arr, v_pts_arr
    cdef n.ndarray[int, ndim=1] tmp_vel_inds
    cdef n.ndarray[double, ndim=1] tmp_data_vals
    cdef object spa_tree, vel_tree

    #### The Cross-Correlation Algorithm:

    # Setting up spatial grids + kd-tree:
    g_x, g_y = n.meshgrid(c_x, c_y)
    s_pts_arr = n.column_stack([g_x.ravel(), g_y.ravel()])
    spa_tree = spatial.cKDTree(s_pts_arr)

    # Setting up velocity grid + kd-tree:
    v_pts_arr = c_vr.reshape(-1, 1)
    vel_tree = spatial.cKDTree(v_pts_arr)

    for i_bin_r in range(n_bin_r):
        rmin = binpts_r[i_bin_r]
        rmax = binpts_r[i_bin_r + 1]

        for i_bin_v in range(n_bin_v):
            vmin = binpts_v[i_bin_v]
            vmax = binpts_v[i_bin_v + 1]

            # Number of Threads
            num_threads = openmp.omp_get_num_threads()

            for i_x in range(n_x):
                c_xi=c_x[i_x]
                for i_y in range(n_y):
                    c_yi=c_y[i_y]

                    tmp_rad_inds_u = set(spa_tree.query_ball_point([c_xi, c_yi], rmax))
                    tmp_rad_inds_d = set(spa_tree.query_ball_point([c_xi, c_yi], rmin))

                    tmp_rad_inds = n.array(list(tmp_rad_inds_u - tmp_rad_inds_d))

                    # 2-D Array of just spatially localized pixels
                    tmp_data_slice = norm_dcube[tmp_rad_inds[:,0], tmp_rad_inds[:,1],:]

                    for i_vr in range(n_vr):
                        c_vri=c_vr[i_vr]

                        tmpweight = weight[i_x, i_y, i_vr]

                        if tmpweight > 0:

                            tmp_vel_inds_u = set(vel_tree.query_ball_point([c_vri], vmax))
                            tmp_vel_inds_d = set(vel_tree.query_ball_point([c_vri], vmin))

                            tmp_vel_inds = n.array(list(tmp_vel_inds_u - tmp_vel_inds_d))

                            tmp_data_vals = tmp_data_slice[:,tmp_vel_inds].flatten()
                            ncell = tmp_data_vals.size

                            if ncell > 0:
                                corr_mean_cube[i_bin_r, i_bin_v] += weight * n.mean(tmp_data_vals)
                                corr_std_cube[i_bin_r, i_bin_v] += weight * n.std(tmp_data_vals)
                                corr_cells[i_bin_r, i_bin_v] += ncell


    #### Finishing Up

    # Number of sampling points:
    cdef int n_samp_pts = n_x * n_y * n_vr
    cdef n.ndarray[double, ndim=2] final_corr, final_err

    final_corr = corr_mean_cube
    final_err = n.sqrt(corr_std_cube)

    # Printing stats on completion 

    cdef object endtime = datetime.datetime.now()
    cdef float time_taken = 0.0
    time_taken = (endtime - starttime).total_seconds()

    print("Number of Threads Used: %i" % num_threads)
    print("Timing: Cross-correlation Finished at %s" % str(endtime))
    print("Timing: The cross-correlation took %1.4f seconds" % time_taken)

    return final_corr, final_err, corr_cells


@cython.wraparound(False)
@cython.boundscheck(False)
def vel_space_corr_bf(n.ndarray[double, ndim=3] dcube, 
                      n.ndarray[double, ndim=1] c_x, 
                      n.ndarray[double, ndim=1] c_y, 
                      n.ndarray[double, ndim=1] c_vr, 
                      n.ndarray[double, ndim=1] binpts_r, 
                      n.ndarray[double, ndim=1] binpts_v,
                      bint norm=False, 
                      double baseline=0.0,
                      bint debug=False):

    cdef int n_bin_r = n.size(binpts_r) - 1
    cdef int n_bin_v = n.size(binpts_v) - 1
    cdef int n_x = dcube.shape[0]
    cdef int n_y = dcube.shape[1]
    cdef int n_vr = dcube.shape[2]
    cdef object starttime = datetime.datetime.now()
 
    # Mean density per spaxel
    # mean_dens = n.mean(dcube, axis=(0, 1)).reshape((1, 1, n_vr))
    cdef double mean_dens = n.mean(dcube - baseline)

    cdef n.ndarray[double, ndim=3] norm_dcube, weight

    if norm:
        norm_dcube = (dcube - mean_dens) / mean_dens
    else:
        norm_dcube = (dcube - baseline) / mean_dens

    # Where the Cross-correlation Mean will go:
    cdef n.ndarray[double, ndim=2] corr_mean_cube = n.zeros(
        (n_bin_r, n_bin_v))

    # Where the number of cells will go:
    cdef n.ndarray[double, ndim=2] corr_cells = n.zeros(
        (n_bin_r, n_bin_v))

    # Where the Standard Deviation will go:
    cdef n.ndarray[double, ndim=2] corr_std_cube = n.zeros(
        (n_bin_r, n_bin_v))

    # The Actual Cross-Correlation:

    # For each cross-correlation bin

    cdef int i_x, i_y, i_vr, i_bin_r, i_bin_v
    cdef double rmin, rmax, vrmin, vrmax
    cdef double c_xi, c_yi, c_vri
    cdef double sumx, sumx2, ncell, meanval, stdval, tmpweight
    cdef double *arr1 

    # Stats on OpenMP threads
    cdef int num_threads, num_proc
    num_proc = openmp.omp_get_num_procs()

    # Determing weighting:
    # Filtering Negative Numbers in weighting:
    weight = n.copy(dcube - baseline)
    weight[n.where(weight < 0)] = 0.0

    weight = weight / n.sum(weight)  # Sum to 1


    print("Number of Procs available: %i" % num_proc)
    print("Timing: Cross-correlation Started at %s" % str(starttime))

    with nogil:
        # Testing the Number of Threads:

        for i_bin_r in prange(n_bin_r):
        # for i_bin_r in prange(n_bin_r, nogil=True):

            # For each radius element
            rmin = binpts_r[i_bin_r]
            rmax = binpts_r[i_bin_r + 1]

            for i_bin_v in prange(n_bin_v):
            # for i_bin_v in prange(n_bin_v, nogil=True):
                num_threads = openmp.omp_get_num_threads()

                # For each velocity element
                vrmin = binpts_v[i_bin_v]
                vrmax = binpts_v[i_bin_v + 1]

                for i_x in range(n_x):
                    c_xi = c_x[i_x]
                    for i_y in range(n_y):
                        c_yi = c_y[i_y]
                        for i_vr in range(n_vr):
                            c_vri = c_vr[i_vr]

                            tmpweight = weight[i_x, i_y, i_vr]

                            if tmpweight > 0:
                                arr1 = cross_corr_2d_1d(
                                    <double*> norm_dcube.data, n_x, n_y, n_vr,
                                    <double*> c_x.data, <double*> c_y.data, 
                                    <double*> c_vr.data, 
                                    c_xi, c_yi, c_vri,
                                    rmin, rmax, vrmin, vrmax 
                                )

                                sumx, sumx2, ncell = arr1[0], arr1[1], arr1[2]

                                if ncell > 0:
                                    meanval = tmpweight * mean(sumx, ncell)
                                    stdval = (tmpweight * stddev(sumx2, ncell, meanval))**2
                                    corr_mean_cube[i_bin_r, i_bin_v] += meanval
                                    corr_std_cube[i_bin_r, i_bin_v] += stdval
                                    corr_cells[i_bin_r, i_bin_v] += ncell


    # Number of sampling points:
    cdef int n_samp_pts = n_x * n_y * n_vr
    cdef n.ndarray[double, ndim=2] final_corr, final_err

    final_corr = corr_mean_cube
    final_err = n.sqrt(corr_std_cube)

    # Printing stats on completion 

    cdef object endtime = datetime.datetime.now()
    cdef float time_taken = 0.0
    time_taken = (endtime - starttime).total_seconds()

    print("Number of Threads Used: %i" % num_threads)
    print("Timing: Cross-correlation Finished at %s" % str(endtime))
    print("Timing: The cross-correlation took %1.4f seconds" % time_taken)

    return final_corr, final_err, corr_cells

#### UTILITY FUNCTIONS ####


def gaussian_function(x, pars):
    return pars[0] * n.exp(-(x - pars[1])**2 / (2 * pars[2]**2))


@cython.boundscheck(False)
def between_1d(n.ndarray[double, ndim=1] arr, double x1, double x2):
    """
    A boolean "between" -- True when between x1 and x2
    """
    cdef n.ndarray[n.uint8_t,cast=True,ndim=1] x = ((arr >= x1) & (arr < x2))

    return x

@cython.boundscheck(False)
def between_2d(n.ndarray[double, ndim=2] arr, double x1, double x2):
    """
    A boolean "between" -- True when between x1 and x2
    """
    cdef n.ndarray[n.uint8_t,cast=True,ndim=2] x = ((arr >= x1) & (arr < x2))

    return x


def between_3d(n.ndarray[double, ndim=3] arr, double x1, double x2):
    """
    A boolean "between" -- True when between x1 and x2
    """
    cdef n.ndarray[n.uint8_t,cast=True,ndim=3] x = ((arr >= x1) & (arr < x2))

    return x


def make_dist_1d(ref_pt, ind_vect):

    ref_grid = n.reshape(ref_pt, (-1, 1)) * n.ones((1, n.size(ind_vect)))
    ind_grid = n.ones((n.size(ref_pt), 1)) * n.reshape(ind_vect, (1, -1))
    dist_vect = n.abs(ref_grid - ind_grid)

    return dist_vect


def make_dist_2d(ref_pt, ind_vect_x, ind_vect_y):
    n_x = n.size(ind_vect_x)
    n_y = n.size(ind_vect_y)
    n_pt = n.shape(ref_pt)[0]

    grid_x, grid_y = n.meshgrid(ind_vect_x, ind_vect_y)

    mgrid_x = n.ones((n_pt, 1, 1)) * grid_x.reshape((1, n_x, n_y))
    mgrid_y = n.ones((n_pt, 1, 1)) * grid_y.reshape((1, n_x, n_y))

    ref_grid = n.reshape(ref_pt, (n_pt, 2, 1, 1)) * n.ones((2, n_x, n_y))

    dist_array = n.sqrt(
        (ref_grid[:, 0, :, :] - mgrid_x)**2 +
        (ref_grid[:, 1, :, :] - mgrid_y)**2)

    return dist_array

@cython.boundscheck(False)
def make_dist_1d_2d(double ref_x, double ref_y, double ref_vr, 
    n.ndarray[double, ndim=1] ind_vect_x, 
    n.ndarray[double, ndim=1] ind_vect_y, 
    n.ndarray[double, ndim=1] ind_vect_vr):

    cdef n.ndarray[double, ndim=3] grid_x, grid_y, grid_vr

    grid_x, grid_y, grid_vr = n.meshgrid(
        ind_vect_x, ind_vect_y, ind_vect_vr)

    cdef n.ndarray[double, ndim=3] dist_array_r = n.sqrt(
        (grid_x - ref_x)**2 +
        (grid_y - ref_y)**2
    )

    cdef n.ndarray[double, ndim=3] dist_array_v = n.abs(grid_vr - ref_vr)

    return dist_array_r, dist_array_v

### Inline Functions

cdef inline double stddev(double sumx2, double ncell, double mean) nogil:
    """
    Calculating the Standard Deviation from mean, sum of x2 and number of cells 
    """
    return ((sumx2 - mean**2)/(ncell))**0.5

cdef inline double mean(double sumx, double ncell) nogil:
    """
    Calculating the mean from sum of x and number of cells 
    """
    return (sumx)/(1.0*ncell)

