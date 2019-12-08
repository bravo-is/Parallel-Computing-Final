/*
MODIFIED FROM CUDA BY EXAMPLE CH.7
Code Sustantially Modified into Conway's Game of Life by
Israel Bravo, Smit Patel, Prathamesh Bramhankar
for UC-Parallel Computing-Fall Semester-2019
*/


#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"
#include <stdlib.h>
#include <time.h>

#define DIM 1024

// these exist on the GPU side
texture<float,2>  texIn;
texture<float,2>  texOut;

__global__ void GOL_kernel( float *dst, bool dstOut ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float   t, l, c, r, b, tl, tr, bl, br, neighbors;
    if (dstOut) {
      t = tex2D(texIn,x,y-1);//top
      l = tex2D(texIn,x-1,y);//left
      c = tex2D(texIn,x,y);//center
      r = tex2D(texIn,x+1,y);//right
      b = tex2D(texIn,x,y+1);//bottom
      tl = tex2D(texIn,x-1,y-1);//top-left
      tr = tex2D(texIn,x+1,y-1);//top-right
      bl = tex2D(texIn,x-1,y+1);//bottom-left
      br = tex2D(texIn,x+1,y+1);//bottom-right
    }else{
      t = tex2D(texOut,x,y-1);//top
      l = tex2D(texOut,x-1,y);//left
      c = tex2D(texOut,x,y);//center
      r = tex2D(texOut,x+1,y);//right
      b = tex2D(texOut,x,y+1);//bottom
      tl = tex2D(texOut,x-1,y-1);//top-left
      tr = tex2D(texOut,x+1,y-1);//top-right
      bl = tex2D(texOut,x-1,y+1);//bottom-left
      br = tex2D(texOut,x+1,y+1);//bottom-right
    }
    neighbors = t+l+r+b+tl+tr+bl+br;
    //Game of Life Rules
    if ( c == 1.0f && neighbors < 2.0f ){
      dst[offset] = 0.0f;
    }
    else if ( c == 1.0f && (neighbors == 2.0f || neighbors == 3.0f) ){
      dst[offset] = 1.0f;
    }
    else if ( c == 1.0f && neighbors > 3.0f ){
      dst[offset] = 0.0f;
    }
    else if ( c == 0.0f && neighbors == 3.0f ){
      dst[offset] = 1.0f;
    }
    else {
      dst[offset] = c;
    }
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    CPUAnimBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

void anim_gpu( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    // we maintain this so that cylce speed can be controlled by timesteps or FPS
    volatile bool dstOut = true;
    for (int i=0; i<2; i++) {
        float *out;
        if (dstOut) {
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
        }
        GOL_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );

    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) {
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    time_t t;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
                                   data.dev_inSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
                                   data.dev_outSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    /* Intializes random number generator */
    srand((unsigned) time(&t));

    // randomly populate the board
    float *cellState = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
      if ( rand() % 2 == 0 ){
        cellState[i] = 0.0f;
      }else{
        cellState[i] = 1.0f;
      }
    }

    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, cellState,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( cellState );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}
