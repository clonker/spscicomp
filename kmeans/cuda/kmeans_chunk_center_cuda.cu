#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#define NO_IMPORT_ARRAY

#include <stdio.h>
#include <stdlib.h>
#include "Python.h"
#include "math.h"
#include "numpy/arrayobject.h"
#include <cuda_runtime.h>
#include <cuda.h>

/* Needs to be compiled as C files because of the Naming problem in Namespace */
#ifdef __cplusplus
extern "C" {
#endif

/*
    Calculate the new centers in the received block.
*/


#define THREAD_NUM   512
#define BLOCK_NUM   128
#define DATA_SIZE  THREAD_NUM*BLOCK_NUM

__global__ void chunk_centers_sum_cuda(double *cu_data,double *cu_centers, int* cu_centers_counter, double* cu_new_centers, int* cu_data_assigns, int* cluster_size,int *dimension)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i,j,k;

    for(k = bid * THREAD_NUM + tid; k < DATA_SIZE; k+= BLOCK_NUM * THREAD_NUM)
    {
        double min_distance = 1E100;
        double distance;
        *(cu_data_assigns+k)=0;
        for (i = 0; i < *cluster_size; i++)
        {
            distance = 0;
            for (j = 0; j < *dimension; j++)
            {
                distance +=(*(cu_data+k*(*dimension)+j)-*(cu_centers+i*(*dimension)+j)) * (*(cu_data+k*(*dimension)+j)-*(cu_centers+i*(*dimension)+j));
            }
            if (distance <= min_distance)
            {
                min_distance = distance;
                *(cu_data_assigns+k) = i;
            }
        }
        __syncthreads();

        if(tid == 0)
        {
            *(cu_centers_counter+bid)=0;
            for (i=0 ; i<THREAD_NUM; i++)
            {
                *(cu_centers_counter+bid*(*cluster_size)+*(cu_data_assigns+bid*THREAD_NUM+i))+=1;
                for (j = 0; j < *dimension; j++)
                {
                    *(cu_new_centers +bid*(*cluster_size)*(*dimension) +(*(cu_data_assigns+bid*THREAD_NUM+i))* (*dimension) + j) += *(cu_data+(bid*THREAD_NUM+i)*(*dimension)+j);
                }
            }
           // printf("%d %d %d\n", *(cu_centers_counter+bid*(*cluster_size)),*(cu_centers_counter+bid*(*cluster_size)+1),*(cu_centers_counter+bid*(*cluster_size)+2));

        }
    }
}


PyObject* kmeans_chunk_center_cuda(PyArrayObject *data, PyArrayObject *centers, PyObject *data_assigns)
{
    /* Record the nearest center of each point and renew the centers with the points near one given center. */
    int cluster_size, dimension, chunk_size;
    cluster_size = *(int *)PyArray_DIMS(centers);
    dimension = PyArray_DIM(centers, 1);
    chunk_size = *(int *)PyArray_DIMS(data);
    int *centers_counter = (int *)malloc(sizeof(int) * BLOCK_NUM* cluster_size);
    double *new_centers = (double *)malloc(sizeof(double)* BLOCK_NUM * cluster_size * dimension);
    int* p_data_assigns= (int *)malloc(sizeof(int) * chunk_size);

    int i, j, k;

    for (i = 0; i < cluster_size* BLOCK_NUM; i++)
    {
	    (*(centers_counter + i)) = 0;
    }

    for (i = 0; i < cluster_size * dimension* BLOCK_NUM; i++)
    {
	    (*(new_centers + i)) = 0;
    }

    double* p_data=(double *)PyArray_DATA(data);
    double* p_centers=(double *)PyArray_DATA(centers);

    double* cu_data, *cu_centers, *cu_new_centers;
    int* cu_centers_counter, *cu_cluster_size, *cu_dimension, *cu_data_assigns;

    cudaMalloc((void**) &cu_data, sizeof(double) * chunk_size * dimension);
    cudaMalloc((void**) &cu_centers, sizeof(double) * cluster_size * dimension);
    cudaMalloc((void**) &cu_centers_counter, sizeof(int) * BLOCK_NUM * cluster_size);
    cudaMalloc((void**) &cu_new_centers, sizeof(double) * BLOCK_NUM * cluster_size * dimension);
    cudaMalloc((void**) &cu_data_assigns, sizeof(int) * chunk_size );
    cudaMalloc((void**) &cu_cluster_size, sizeof(int) *1);
    cudaMalloc((void**) &cu_dimension, sizeof(int) *1);

    cudaMemcpy(cu_data, p_data, sizeof(double) * chunk_size * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_centers, p_centers, sizeof(double) * cluster_size * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_centers_counter, centers_counter, sizeof(int)* BLOCK_NUM * cluster_size,cudaMemcpyHostToDevice);
    cudaMemcpy(cu_new_centers, new_centers, sizeof(double) * BLOCK_NUM * cluster_size * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_data_assigns, p_data_assigns, sizeof(int) *chunk_size , cudaMemcpyHostToDevice);
    cudaMemcpy(cu_cluster_size, &cluster_size, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_dimension, &dimension, sizeof(int) * 1, cudaMemcpyHostToDevice);


    chunk_centers_sum_cuda<<<BLOCK_NUM, THREAD_NUM>>>(cu_data,cu_centers,cu_centers_counter,cu_new_centers,cu_data_assigns,cu_cluster_size,cu_dimension);

    cudaMemcpy(centers_counter, cu_centers_counter,sizeof(int) * BLOCK_NUM *cluster_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_centers, cu_new_centers, sizeof(double) * BLOCK_NUM* cluster_size * dimension, cudaMemcpyDeviceToHost);
    cudaMemcpy(p_data_assigns, cu_data_assigns, sizeof(int) * chunk_size  , cudaMemcpyDeviceToHost);

    for (i=0; i<BLOCK_NUM ;i++)
    {
        for (j=0;j<cluster_size; j++)
        {
            *(centers_counter+j)+=*(centers_counter+i*cluster_size+j);
        }
    }
    for (i=0; i<BLOCK_NUM ;i++)
    {
        for (j=0;j<cluster_size; j++)
        {
            for (k=0;k<dimension;k++)
            *(new_centers+j*dimension+k)+=*(new_centers+i*cluster_size*dimension+j*dimension+k);
        }
    }

    cudaFree(cu_data);
    cudaFree(cu_centers);
    cudaFree(cu_centers_counter);
    cudaFree(cu_new_centers);
    cudaFree(cu_data_assigns);
    cudaFree(cu_cluster_size);
    cudaFree(cu_dimension);


    for (i = 0; i < cluster_size; i++)
    {
        if (*(centers_counter + i) == 0)
        {
            for (j = 0; j < dimension; j++)
            {
                (*(new_centers + i * dimension + j)) = (*(double*)PyArray_GETPTR2(centers, i, j));
            }
        }
        else
        {
            for (j=0; j < dimension; j++)
            {
                (*(new_centers + i * dimension + j)) /= (*(centers_counter + i));
                //printf("%lf ",(*(new_centers + i * dimension + j)) );
            }
        }
    }

    for (i=0; i<chunk_size; i++)
    {
        PyList_SetItem(data_assigns, i, PyInt_FromLong(*(p_data_assigns+i)));
    }

    PyObject* return_new_centers;
    npy_intp dims[2] = {cluster_size, dimension};
    return_new_centers = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    void *arr_data = PyArray_DATA((PyArrayObject*)return_new_centers);
    memcpy(arr_data, new_centers, PyArray_ITEMSIZE((PyArrayObject*) return_new_centers) * cluster_size * dimension);
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    free(centers_counter);
    free(new_centers);
    free(p_data_assigns);
    return (PyObject*) return_new_centers;
}

#ifdef __cplusplus
}
#endif
