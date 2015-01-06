#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

static int get_closest_center(int gid, __global const float *data, __global const float *centers, int dim, int n_centers) {
    int minCenter = 0;
    float minDist = INFINITY, currDist;
    for(int c = 0; c < n_centers; c++) {
        currDist = 0;
        for (int i = 0; i < dim; i++) {
            float xi = data[dim*gid + i];
            float ci = centers[dim*c + i];
            currDist += (xi-ci)*(xi-ci);
        }
        currDist = sqrt(currDist);
        if(currDist < minDist) {
            minDist = currDist;
            minCenter = c;
        }
    }
    return minCenter;
}

/*void atomic_add_local(volatile local float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;

    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}*/


__kernel void kmeans_chunk_center_cl(
    __global const float *data,
    __global const float *centers,
    __global int *assigns,
    __global float *newCenters,
    __global int* centersCounter,
    int dim,
    int n_centers
) {

    int gid = get_global_id(0);

    __local int tmp_centersCounter[sizeof(centersCounter) / sizeof(int)];
    //__local float tmp_newCenters[sizeof(newCenters)];
    for(int k = 0; k < n_centers; k++) {
        tmp_centersCounter[k] = 0;
        for(int i = 0; i < dim; i++) {
            //tmp_newCenters[dim*k+i] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate closest center to the current data point
    int closestCenter = get_closest_center(gid, data, centers, dim, n_centers);

    // remember the mapping of data point -> center
    assigns[gid] = closestCenter;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&tmp_centersCounter[closestCenter], 1);
    for(int i = 0; i < dim; i++) {
        //atomic_add_local(&tmp_newCenters[dim*closestCenter+i], data[dim*gid + i]);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if ( get_local_id(0) == get_local_size(0) - 1 ) {
        for(int k = 0; k < n_centers; k++) {
            atomic_add(&centersCounter[k], tmp_centersCounter[k]);
            for(int i = 0; i < dim; i++) {
                //atomic_add_global(&newCenters[dim*k+i], tmp_newCenters[dim*k+i]);
            }
        }
    }
}