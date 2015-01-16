#define DIM ${DIM}
#define K ${K}

kernel void kmeans_chunk_center_cl(
    global const float* data,
    global int* assigns,
    global const float* centers,
    global int* centersCounter,
    global float* newCenters,
    global float* out
) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    int minCenter = 3;
    float minDist = INFINITY;
    for(int c = 0; c < K; c++) {
        float currDist = 0;
        for (int i = 0; i < DIM; i++) {
            float xi = data[DIM*gid + i];
            float ci = centers[DIM*c + i];
            currDist += (xi-ci)*(xi-ci);
        }
        currDist = sqrt(currDist);
        if(currDist < minDist) {
            minDist = currDist;
            minCenter = c;
        }
    }
    minCenter = 3;
    int closestCenter = minCenter;
    out[gid] = closestCenter;
    assigns[gid] = closestCenter;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if(lid == 0) {
        unsigned int offset = get_group_id(0) * get_local_size(0);
        for(int k = 0; k < K; k++) {
            centersCounter[offset + k] = 0;
            for(int d = 0; d < DIM; d++) {
                newCenters[offset + k + d] = 0.0f;
            }
        }
        out[0] = newCenters[0];
        out[1] = newCenters[1];
        for(unsigned int k = 0; k < get_local_size(0); k++) {
            int assign = assigns[offset+k];
            centersCounter[assign + offset] += 1;
            for(int d = 0; d < DIM; d++) {
                newCenters[offset + assign + d] += data[offset+k+d];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (gid == 0) {
        for(int i = 0; i < K; i++) {
            centersCounter[i] += centersCounter[i + get_local_size(0)];
            for(uint g = 1; g < get_num_groups(0); g++) {
                for(int d = 0; d < DIM; d++) {
                    newCenters[i+d] += newCenters[i + g + get_local_size(0)];
                }
            }
        }

        for(int k=0; k < K; k++) {
            if(centersCounter[k] > 0) {
                for(int d = 0; d < DIM; d++) {
                    newCenters[k+d] /= (float) centersCounter[k];
                }
            } else {
                for(int d = 0; d < DIM; d++) {
                    newCenters[k+d] = centers[k+d];
                }
            }
        }
        out[2] = newCenters[0];
        out[3] = newCenters[1];
        out[0] = 1234567;
    }
}