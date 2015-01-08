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

    __local int tmp_centersCounter[sizeof(centersCounter)];
    for(int k = 0; k < n_centers; k++) {
        tmp_centersCounter[k] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate closest center to the current data point
    int closestCenter = get_closest_center(gid, data, centers, dim, n_centers);

    // remember the mapping of data point -> center
    assigns[gid] = closestCenter;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_inc(&tmp_centersCounter[closestCenter]);

    barrier(CLK_GLOBAL_MEM_FENCE);
    if ( get_local_id(0) == get_local_size(0) - 1 ) {
        for(int k = 0; k < n_centers; k++) {
            if(tmp_centersCounter[k] > 0) {
                atomic_add(&centersCounter[k], tmp_centersCounter[k]);
            } else {
                centersCounter[k] = 0;
            }
        }
    }
}