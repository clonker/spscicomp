#define N ${N}
#define M ${M}

#define ALPHA(x,y) alpha[(x)*N + y]
#define BETA(x,y) beta[(x)*N + y]
#define GAMMA(x,y) gamma[(x)*N + y]
#define WEIGHTS(x) weights[x]

#define A(i,j) a[(i)*N + j]
#define B(i,k) b[(i)*M + k]
#define PI(i) pi[i]
#define O(t) obs[(t)]
#define XI(t,i,j) xi[(t)*N*N + (i)*N + j]
#define RXI(t,i,j) rxi[(t)*N*N + (i)*N + j]
#define SCRATCH(t,i,j) scratch[(t)*N*N + (i)*N + j]

#define B_COUNT(t,i,j) bc[(t)*N*M + (i)*M + j]
#define RB(t,i,j) rb[(t)*N*M + (i)*M + j]
#define C(t,i,j) c[(t)*N*N + (i)*N + j]

int
is_last_in_time(long gid, __constant long *T, int K)
{
    for (int k = 1; k <= K; k++) {
        if (gid == T[k]-1)
            return 1;
    }
    return 0;
}

int
k_from_gid(long gid, __constant long *T, int K)
{
    for (int k = 1; k <= K; k++)
        if (gid < T[k])
            return k-1;
    return -1;
}

__kernel
void prepare_backward(
        __global float *a,
        __global float *b,
        __global int *obs,
        __global float *c,
        __constant long *T, int K)
{
    size_t gid = get_global_id(0);
    float p_C[N][N] = { 0.0f };
    float p_B[N];
    float p_A[N][N];

    for (int i = 0; i < N; i++)
        p_B[i] = B(i,O(gid+1));

    if (is_last_in_time(gid, T, K)) {
        for (int i = 0; i < N; i++)
            p_C[i][i] = 1;

    } else {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                p_A[i][j] = A(i,j);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                p_C[i][j] = p_A[i][j]*p_B[j];
    }
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C(gid, i, j) = p_C[i][j];
}

__kernel
void prepare_forward(
        __global float *a,
        __global float *b,
        __global float *pi,
        __global int *obs,
        __global float *c)
{
    size_t gid = get_global_id(0);
    float p_C[N][N] = { 0.0f };
    float p_B[N];
    float p_A[N][N];
    float p_pi[N];

    for (int i = 0; i < N; i++)
        p_B[i] = B(i,O(gid));

    if (gid == 0) {
        for (int i = 0; i < N; i++)
            p_pi[i] = PI(i);
        for (int i = 0; i < N; i++)
            p_C[i][i] = p_B[i]*p_pi[i];

    } else {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                p_A[i][j] = A(i,j);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                p_C[i][j] = p_A[j][i]*p_B[i];
    }
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C(gid, i, j) = p_C[i][j];
}


__kernel
void make_gamma(
        __global float *alpha,
        __global float *beta,
        __global float *gamma)
{
    int gid = get_global_id(0);
    float a_t[N];
    float b_t[N];
    // get data from global mem
    for (int i = 0; i < N; i++) {
        a_t[i] = ALPHA(gid,i);
        b_t[i] = BETA(gid,i);
    }

    // compute gamma_t
    float gamma_t[N]  = { 0.0f };
    float gamma_t_sum =   0.0f  ;
    for (int i = 0; i < N; i++) {
        gamma_t[i]   = a_t[i] * b_t[i];
        gamma_t_sum += gamma_t[i];
    }
    for (int i = 0; i < N; i++) {
        gamma_t[i] /= gamma_t_sum;
        GAMMA(gid,i) = gamma_t[i];
    }
}

__kernel
void make_xi(
        __global float *a,
        __global float *b,
        __global float *alpha,
        __global float *beta,
        __global float *xi,
        __global float *weights,
        __global int *obs,
        __constant long *T, int K)
{
    size_t gid = get_global_id(0);
//    int k = k_from_gid(gid, T, K);
//    float weight = weights[k];

    if (!is_last_in_time(gid, T, K)) {

        float xi_t[N][N] = { 0.0f };
        float xi_t_sum   =   0.0f  ;    
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                xi_t[i][j] = ALPHA(gid,i)*A(i,j)*B(j,O(gid+1))*BETA(gid+1,j);
                xi_t_sum  += xi_t[i][j];
            }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                xi_t[i][j] = xi_t[i][j] / xi_t_sum;
                XI(gid, i, j) = xi_t[i][j];
            }
    }
}

__kernel
void make_bc(
        __global float *gamma,
        __global float *bc,
        __global int *obs)
{
    long gid = get_global_id(0);
    float gamma_t[N] = { 0.0f };

   for (int i = 0; i < N; i++)
        gamma_t[i] = GAMMA(gid, i);
    int ob = O(gid);

    for (int i = 0; i < N; i++)
        for (int k = 0; k < M; k++) {
            if (ob == k)
                B_COUNT(gid, i, k) = gamma_t[i];
            else
                B_COUNT(gid, i, k) = 0.0f;
        } 
}

__kernel
void reduce_bc(
        __global float *bc,
        __global float *rb,
        __local float *scratch,
        long len)
{
    long gid = get_global_id(0);
    float b_sum[N][M] = { 0.0f };
    
     // loop sequentually over the chunks of time
    while (gid < len) {
        for (int i = 0; i < N; i++)
            for (int k = 0; k < M; k++)
                b_sum[i][k] += B_COUNT(gid, i, k);
        gid += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            SCRATCH(local_index, i, j) = b_sum[i][j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = get_local_size(0)/2;
              offset > 0;
              offset >>= 1) {
        if (local_index < offset)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                    SCRATCH(local_index,i,j) += SCRATCH(local_index+offset,i,j);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                RB(get_group_id(0), i, j) = SCRATCH(0, i, j);
   }   
}

__kernel
void reduce_xi(
        __global float *xi,
        __global float *rxi,
        __local  float *scratch,
        __constant long *T, int K)
{
    long gid = get_global_id(0);
    float xi_sum[N][N] = { 0.0f };

    // loop sequentually over the chunks of time
    while (gid < T[K]) {
        if (!is_last_in_time(gid, T, K))
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    xi_sum[i][j] += XI(gid, i, j);
        gid += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            SCRATCH(local_index, i, j) = xi_sum[i][j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = get_local_size(0)/2;
              offset > 0;
              offset >>= 1) {
        if (local_index < offset)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    SCRATCH(local_index,i,j) += SCRATCH(local_index+offset,i,j);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                RXI(get_group_id(0), i, j) = SCRATCH(0, i, j);
   }
}

__kernel
void reduce_forward(
        __global float *c,
        __global float *alpha,
        __global float *weights)
{
    size_t global_index = get_global_id(0);

    if (global_index == 0) {
        float alpha_0[N];
        float weight = 0.0f;
        for (int i = 0; i < N; i++) {
            alpha_0[i] = C(0,i,i);
            weight += alpha_0[i];
        }
        for (int i = 0; i < N; i++) {
            alpha_0[i] /= weight;
            ALPHA(0,i) = alpha_0[i];
            WEIGHTS(0) = weight;
        }
    }

    for (size_t offset = 1;
                 offset < get_global_size(0);
                 offset <<= 1) {
        size_t done = offset<<1;
        float C_t[N][N] = { 0.0f };
        if (global_index >= offset) {
            if (global_index >= done) {
                // get matrices from global memory
                float C_1[N][N], C_2[N][N];
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        C_1[i][j] = C(global_index, i, j);
                        C_2[i][j] = C(global_index - offset, i, j);
                    }
                // multiply matrices
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        for (int k = 0; k < N; k++)
                            C_t[i][j] += C_1[i][k]*C_2[k][j];
            } else {
                // matrix and alpha_t
                float alpha_old[N];
                float alpha_new[N] = { 0.0f };
                float weight = 0.0f;
                for (int i = 0; i < N; i++) {
                    alpha_old[i] = ALPHA(global_index - offset, i);
                    for (int j = 0; j < N; j++)
                        C_t[i][j] = C(global_index, i, j);
                }
                // compute matrix*vector
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        alpha_new[i] += C_t[i][j]*alpha_old[j];
                    weight += alpha_new[i];
                }
                for (int i = 0; i < N; i++) {
                    alpha_new[i] /= weight;
                    ALPHA(global_index, i) = alpha_new[i];
                    WEIGHTS(global_index) = weight;
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (global_index >= done)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    C(global_index, i, j) = C_t[i][j];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel
void reduce_backward(
        __global float *c,
        __global float *beta,
        __global float *weights,
        __constant long *T, int K)
{
    size_t global_index = get_global_id(0);

    if (is_last_in_time(global_index, T, K)) {
        for (int i = 0; i < N; i++) {
            BETA(global_index,i) = C(global_index,i,i)*WEIGHTS(global_index);
        }
    }
    for (size_t offset = 1;
                 offset < get_global_size(0);
                 offset <<= 1) {
        size_t done = offset<<1;
        float C_t[N][N] = { 0.0f };
        if ((global_index + offset) < get_global_size(0)) {
            if ((global_index + done) < get_global_size(0)) {
                // get matrices from global memory
                float C_1[N][N];
                float C_2[N][N];
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        C_1[i][j] = C(global_index, i, j);
                        C_2[i][j] = C(global_index + offset, i, j);
                    }
                // multiply matrices
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        for (int k = 0; k < N; k++)
                            C_t[i][j] += C_1[i][k]*C_2[k][j];
            } else {
                // matrix and beta_t+1
                float beta_old[N];
                float beta_new[N] = { 0.0f };
                for (int i = 0; i < N; i++) {
                    beta_old[i] = BETA(global_index + offset, i);
                    for (int j = 0; j < N; j++)
                        C_t[i][j] = C(global_index, i, j);
                }
                // compute matrix*vector
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) 
                        beta_new[i] += C_t[i][j]*beta_old[j];
                }
                for (int i = 0; i < N; i++)
                    BETA(global_index, i) = beta_new[i] / WEIGHTS(global_index);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if ((global_index + done) <= get_global_size(0))
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    C(global_index, i, j) = C_t[i][j];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
