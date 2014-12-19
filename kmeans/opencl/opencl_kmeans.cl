__kernel void kmeans_chunk_center_cl(__global const float *data, __global const float *centers, __global int *assigns, __global float *newCenters, __global int *centersCounter, __global int *dim, __global int *ncenters) {
    int gid = get_global_id(0);

    int minCenter = 0;
    float minDist = INFINITY;
    float currDist = 0;
    for(int c = 0; c < ncenters[0]; c++) {
        for (int i = 0; i < dim[0]; i++) {
            float xi = data[dim[0]*gid + i];
            float ci = centers[dim[0]*c + i];
            int minCenter = 0;
            currDist = xi*xi + ci*ci;
        }
        currDist = sqrt(currDist);
        if(currDist < minDist) {
            minDist = currDist;
            minCenter = c;
        }
    }

    assigns[gid] = minCenter;

    //barrier(CLK_GLOBAL_MEM_FENCE);

    /*centersCounter[minCenter] += 1;
    for (int i = 0; i < dim[0]; i++) {
        float xi = data[dim[0]*gid + i];
        newCenters[dim[0]*minCenter + i] = xi;
    }*/
}