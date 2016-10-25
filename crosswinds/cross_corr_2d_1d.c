// 
// The underlying brute-force cross-correlation algorithm for the
// 2-D (Spatial) + 1-D (Velocity) Cross-correlation. 
// 
// 

#include <stdlib.h>
#include <math.h>


double dist2d_c(double x1, double y1, double x2, double y2)
{
    return sqrt(((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2)));
}

double dist1d_c(double x1, double x2)
{
     return fabs(x1 - x2);
}   

// 
// The core cross-correlation function
//
double *cross_corr_2d_1d(double* norm_dcube, 
    int n_x, int n_y, int n_vr,
    double* c_x, double* c_y, 
    double* c_vr,
    double c_xi, double c_yi, double c_vri,
    double rmin, double rmax, double vrmin, double vrmax 
    )
{
    // Creating object for return values:
    double* vals;
    vals = malloc(sizeof(double)*3);

    // All variables being defined early
    double sumx = 0.0, sumx2 = 0.0, ncell = 0.0, tmpval = 0.0;
    double tmp_x, tmp_y, tmp_vr, rdist, vrdist;
    int i_x, i_y, i_vr;

    for (i_x = 0; i_x<n_x; i_x++)
    {
        tmp_x = c_x[i_x];

        for (i_y = 0 ; i_y<n_y; i_y++)
        {
            tmp_y = c_y[i_y];
            rdist = dist2d_c(tmp_x, tmp_y, c_xi, c_yi);


            if ((rdist >= rmin ) && (rdist < rmax))
            {

                for (i_vr = 0; i_vr<n_vr; i_vr++)
                {
                    tmp_vr = c_vr[i_vr];
                    vrdist = dist1d_c(tmp_vr, c_vri);


                    if ((vrdist >= vrmin ) && (vrdist < vrmax))
                    {

                        tmpval = norm_dcube[(n_y*n_vr*i_x) + (n_vr*i_y) + i_vr];
                        sumx += tmpval;
                        sumx2 += tmpval*tmpval;
                        ncell += 1.0;
                    }
                }
            }
        }
    }

    vals[0] = sumx;
    vals[1] = sumx2;
    vals[2] = ncell;

    // printf("%f\n", vals[0]);
    // printf("%f\n", vals[1]);
    // printf("%f\n", ncell);

    return vals;
}
