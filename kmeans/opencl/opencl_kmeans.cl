#ifndef __DIMS__
#define __DIMS__

#define DIM ${DIM}
#define K ${K}

#endif

__kernel void kmeans_chunk_center_cl(
    __global int* assigns,
    __global const float* data,
    __global const float* centers,
    __global int* centersCounter,
    __global float* newCenters,
    __global float* out
) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    assigns[gid - get_global_offset(0)] = 42;

    int closestCenter = 0;
    float minDist = INFINITY;
    for(int k = 0; k < K; k++) {
        float currDist = 0;
        for (int d = 0; d < DIM; d++) {
            float xi = data[DIM*(gid - get_global_offset(0)) + d];
            float ci = centers[DIM*k + d];
            currDist += (xi-ci)*(xi-ci);
        }
        currDist = sqrt(currDist);
        if(currDist < minDist) {
            minDist = currDist;
            closestCenter = k;
        }
    }
    assigns[gid - get_global_offset(0)] = closestCenter;
    out[gid%6] = (float) closestCenter;

    /*
        global_id(d) = global_offset(d) + local_id(d) + group_id(d) * local_size(d)
        global_size(d) = local_size(d) * num_groups(d)
    */
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(lid == get_local_size(0) - 1) {
        unsigned int centerOffset = get_group_id(0) * K;
        unsigned int dataOffset = get_group_id(0)*get_local_size(0);

        for(int k = 0; k < K; k++) {
            centersCounter[centerOffset + k] = 0;
            for(int d = 0; d < DIM; d++) {
                newCenters[DIM*centerOffset + DIM*k + d] = 0.0f;
            }
        }

        for(unsigned int k = 0; k < get_local_size(0); k++) {
            int assign = assigns[dataOffset+k];
            centersCounter[assign + centerOffset] += 1;
            for(int d = 0; d < DIM; d++) {
                newCenters[DIM*centerOffset + DIM*assign + d] += data[DIM*dataOffset + DIM*k + d];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (gid - get_global_offset(0) == 0) {
        for(int i = 0; i < K; i++) {
            for(unsigned int g = 1; g < get_num_groups(0); g++) {
                centersCounter[i] += centersCounter[i + g*K];
                for(int d = 0; d < DIM; d++) {
                    newCenters[DIM*i+d] += newCenters[DIM*i + d + DIM*g*K];
                }
            }
        }

        for(int k=0; k < K; k++) {
            if(centersCounter[k] > 0) {
                for(int d = 0; d < DIM; d++) {
                    newCenters[DIM*k+d] /= (float) centersCounter[k];
                }
            } else {
                for(int d = 0; d < DIM; d++) {
                    newCenters[DIM*k+d] = centers[DIM*k+d];
                }
            }
        }
    }
}