#ifndef __DIMS__
#define __DIMS__

#define DIM ${DIM}
#define K ${K}

#endif

__kernel void kmeans_chunk_center_cl(
    __global const float* data,
    __global int* assigns,
    __global const float* centers,
    __global int* centersCounter,
    __global float* newCenters,
    __global float* out
) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    int closestCenter = 0;
    float minDist = INFINITY;
    for(int k = 0; k < K; k++) {
        float currDist = 0;
        for (int d = 0; d < DIM; d++) {
            float xi = data[DIM*gid + d];
            float ci = centers[DIM*k + d];
            currDist += (xi-ci)*(xi-ci);
        }
        currDist = sqrt(currDist);
        if(currDist < minDist) {
            minDist = currDist;
            closestCenter = k;
        }
    }
    assigns[gid] = closestCenter;

    if (closestCenter > 10000 || closestCenter < 10000) {
        out[0] = 123123123123;
        out[1] = 32132132131;
    }


    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(lid == 0) {
        unsigned int offset = get_group_id(0) * get_local_size(0);
        for(int k = 0; k < K; k++) {
            centersCounter[offset + k] = 0;
            for(int d = 0; d < DIM; d++) {
                newCenters[DIM*offset + DIM*k + d] = 0.0f;
            }
        }

        for(unsigned int k = 0; k < get_local_size(0); k++) {
            int assign = assigns[offset+k];
            centersCounter[assign + offset] += 1;
            for(int d = 0; d < DIM; d++) {
                newCenters[DIM*offset + DIM*assign + d] += data[DIM*offset+DIM*k+d];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (gid == 0) {
        for(int i = 0; i < K; i++) {
            for(unsigned int g = 1; g < get_num_groups(0); g++) {
                centersCounter[i] += centersCounter[i + g*get_local_size(0)];
                for(int d = 0; d < DIM; d++) {
                    newCenters[DIM*i+d] += newCenters[DIM*i + d + DIM*g*get_local_size(0)];
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