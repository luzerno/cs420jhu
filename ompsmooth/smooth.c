/*******************************************************************************
*
*  Driver routine to operate a kernal smoother.
*
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ompsmooth.h"
#include "omp.h"

#define MAT_DIM 6000
#define MAT_SIZE MAT_DIM*MAT_DIM

#define KERNEL_HALFWIDTH 3


int main()
{
  /* Variables for timing */
  struct timeval ta, tb;

  /* Create two input matrixes */
  float * m1in;
  float * m2out;
  float * m3in;
  float * m4out;

  m1in = malloc ( sizeof(float)*MAT_SIZE );
  m2out = malloc ( sizeof(float)*MAT_SIZE );
  m3in = malloc (sizeof(float) *MAT_SIZE);
  m4out = malloc (sizeof(float) *MAT_SIZE);



  /*  random data for the input */
  // int fd = open ("/dev/urandom", O_RDONLY );
  //  read ( fd, m1in, MAT_SIZE*sizeof(float));
  /* instead, this pattern is preserved by the kernel smoother */
  int64_t x, y;
  for (y=0; y<MAT_DIM; y++) {
    for (x=0; x<MAT_DIM; x++) {
      m1in[y*MAT_DIM+x] = (float)(x+y);
      m3in[y*MAT_DIM+x] = (float)(x+y);
    }
  }      

  /* zero the output */
  memset ( m2out, 0, MAT_SIZE*sizeof(float) );
  memset ( m4out, 0, MAT_SIZE*sizeof(float) );


  /*********  Serial Test **********/


  /* run once to warm up the cache before measuring */
  smoothSerialYX ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

  /* get initial time */
  gettimeofday ( &ta, NULL );

  smoothSerialYX ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

  /* get initial time */
  gettimeofday ( &tb, NULL );

  /* Work out the time */
  int s = tb.tv_sec - ta.tv_sec;
  int u;  
  if ( ta.tv_usec <= tb.tv_usec ) {
    u = tb.tv_usec - ta.tv_usec;
  } else {
    u = 1000000 + tb.tv_usec - ta.tv_usec;
    s = s-1;
  }

  printf ("%d\t%d\n",s,u );

  /* get initial time */
  gettimeofday ( &ta, NULL );

  smoothSerialXY ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

  /* get initial time */
  gettimeofday ( &tb, NULL );

  /* Work out the time */
  s = tb.tv_sec - ta.tv_sec;
  u;  
  if ( ta.tv_usec <= tb.tv_usec ) {
    u = tb.tv_usec - ta.tv_usec;
  } else {
    u = 1000000 + tb.tv_usec - ta.tv_usec;
    s = s-1;
  }

  printf ("%d\t%d\n",s,u );

  /*********  Parallel Tests **********/
  int threads;
  for (threads=1; threads <=32; threads*=2)
  {

    omp_set_num_threads(threads); 
    printf("Threads = %d\n", threads);

    /* get initial time */
    gettimeofday ( &ta, NULL );

    smoothParallelYXFor ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

    /* get initial time */
    gettimeofday ( &tb, NULL );

    /* Work out the time */
    s = tb.tv_sec - ta.tv_sec;
    if ( ta.tv_usec < tb.tv_usec ) {
      u = tb.tv_usec - ta.tv_usec;
    } else {
      u = 1000000 + tb.tv_usec - ta.tv_usec;
      s = s-1;
    }

    printf ("%d\t%d\n",s,u );

    /* get initial time */
    gettimeofday ( &ta, NULL );

    smoothParallelXYFor ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

    /* get initial time */
    gettimeofday ( &tb, NULL );

    /* Work out the time */
    s = tb.tv_sec - ta.tv_sec;
    if ( ta.tv_usec < tb.tv_usec ) {
      u = tb.tv_usec - ta.tv_usec;
    } else {
      u = 1000000 + tb.tv_usec - ta.tv_usec;
      s = s-1;
    }

    printf ("%d\t%d\n",s,u );


    /* get initial time */
    gettimeofday ( &ta, NULL );

    smoothParallelCoalescedFor ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out );

    /* get initial time */
    gettimeofday ( &tb, NULL );

    /* Work out the time */
    s = tb.tv_sec - ta.tv_sec;
    if ( ta.tv_usec < tb.tv_usec ) {
      u = tb.tv_usec - ta.tv_usec;
    } else {
      u = 1000000 + tb.tv_usec - ta.tv_usec;
      s = s-1;
    }

    printf ("%d\t%d\n",s,u );

  /* get initial time */
    gettimeofday ( &ta, NULL );

    smoothParallelYXFor2Invocation ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out, m3in, m4out );

    /* get initial time */
    gettimeofday ( &tb, NULL );

    /* Work out the time */
    s = tb.tv_sec - ta.tv_sec;
    if ( ta.tv_usec < tb.tv_usec ) {
      u = tb.tv_usec - ta.tv_usec;
    } else {
      u = 1000000 + tb.tv_usec - ta.tv_usec;
      s = s-1;
    }
    printf ("%d\t%d\n",s,u );
/* get initial time */
    gettimeofday ( &ta, NULL );

    smoothParallelYXForMerge ( MAT_DIM, KERNEL_HALFWIDTH, m1in, m2out, m3in, m4out );

    /* get initial time */
    gettimeofday ( &tb, NULL );

    /* Work out the time */
    s = tb.tv_sec - ta.tv_sec;
    if ( ta.tv_usec < tb.tv_usec ) {
      u = tb.tv_usec - ta.tv_usec;
    } else {
      u = 1000000 + tb.tv_usec - ta.tv_usec;
      s = s-1;
    }
    printf ("%d\t%d\n",s,u );
  }
}

