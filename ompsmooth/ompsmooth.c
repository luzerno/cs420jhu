/*******************************************************************************
*
*  OpenMP Parallel Routines
*
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "ompsmooth.h"

// remove 
# include "assert.h"

/*------------------------------------------------------------------------------
* Name:  evaluate
* Action:  determine the smoothed value
*-----------------------------------------------------------------------------*/
inline float evaluate ( int dim, int halfwidth, int x, int y, float* m1)
{

  float value =0.0;
  int cellcount=0;

  /* kernal iteration variables */
  int xk, yk;

  /* Evaluate the kernel */
  for (yk=-1*halfwidth;yk<=halfwidth;yk++) 
  {
    for (xk=-1*halfwidth;xk<=halfwidth;xk++) 
    {
      /* For each valid cell inside the domain */
      if ((y+yk>=0) && (x+xk>=0) && (y+yk<dim) && (x+xk<dim))
      {
        /* Add the contribution of the cell */
        cellcount++;
        value += m1[dim*(y+yk)+(x+xk)];
      }
    }
  }
  /* Determine the average/smoothed value */
  return value/cellcount;

  /* Check the output.  Turn off for timing rns. */
  //float val = value / cellcount;
  //printf("XY %d, %d cellcount %d value %f\n", x,y,cellcount, value);
}


/*------------------------------------------------------------------------------
* Name:  smoothSerial
* Action:  serial smoother in C (row major) order
*-----------------------------------------------------------------------------*/
void smoothSerialYX ( int dim, int halfwidth, float * m1, float * m2 )
{
  int x,y;
  for (y=0; y<dim; y++)
  {
    for (x=0; x<dim; x++)
    {
      m2[y*dim+x] = evaluate ( dim, halfwidth, x, y, m1 );
    }
  }
}

/*------------------------------------------------------------------------------
* Name:  smoothSerialXY
* Action:  serial smoother in Fortran (column major) order
*-----------------------------------------------------------------------------*/
void smoothSerialXY ( int dim, int halfwidth, float * m1, float * m2 )
{
  int x,y;
  for (x=0; x<dim; x++)
  {
    for (y=0; y<dim; y++)
    {
      m2[y*dim+x] = evaluate ( dim, halfwidth, x, y, m1 );
    }
  }
}


/*------------------------------------------------------------------------------
* Name:  smoothParallelYXFor
* Action: for loop in C (row major) order
*-----------------------------------------------------------------------------*/
void smoothParallelYXFor ( int dim, int halfwidth, float * m1, float * m2 )
{
    int x, y;
    #pragma omp parallel for private (x, y)
    for (y = 0; y < dim; y++) {
        for (x = 0; x < dim; x++) {
            m2[y * dim + x] = evaluate(dim, halfwidth, x, y, m1);
        }
    }

}

/*------------------------------------------------------------------------------
* Name:  smoothParallelXYFor
* Action: for loop in Fortran (column major) order
*-----------------------------------------------------------------------------*/
void smoothParallelXYFor ( int dim, int halfwidth, float * m1, float * m2 )
{
    int x, y;
    #pragma omp parallel for private (x, y)
    for (x = 0; x < dim; x++) {
        for (y = 0; y < dim; y++) {
            m2[y * dim + x] = evaluate(dim, halfwidth, x, y, m1);
        }
    }
}


/*------------------------------------------------------------------------------
* Name:  smoothParallelCoalescedFor
* Action: flattened for loop in C (row major) order
*-----------------------------------------------------------------------------*/
void smoothParallelCoalescedFor ( int dim, int halfwidth, float * m1, float * m2 )
{
    int xy, x, y;
    #pragma omp parallel for private (xy, x, y)
    for (xy = 0; xy < dim * dim; xy++) {
        x = xy / dim;
        y = xy % dim;
        m2[xy] = evaluate(dim, halfwidth, x, y, m1);
    }
}

void smoothParallelYXFor2Invocation(int dim, int halfwidth, float *m1, float *m2, float *m3, float *m4) {
    int x, y;
    #pragma omp parallel for private (x, y)
    for (y = 0; y < dim; y++) {
        for (x = 0; x < dim; x++) {
            m2[y * dim + x] = evaluate(dim, halfwidth, x, y, m1);
        }
    }
    #pragma omp parallel for private (x, y)
    for (y = 0; y < dim; y++) {
        for (x = 0; x < dim; x++) {
            m4[y * dim + x] = evaluate(dim, halfwidth, x, y, m3);
        }
    }
}
void smoothParallelYXForMerge(int dim, int halfwidth, float *m1, float *m2, float *m3, float *m4) {
    int x, y;
    #pragma omp parallel for private (x, y)
    for (y = 0; y < dim; y++) {
        for (x = 0; x < dim; x++) {
            m2[y * dim + x] = evaluate(dim, halfwidth, x, y, m1);
            m4[y * dim + x] = evaluate(dim, halfwidth, x, y, m3);
        }
    }
}

