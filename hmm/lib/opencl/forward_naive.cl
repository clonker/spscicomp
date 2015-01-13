/**
 * parallel GPU optimized kernel implementation of the forward algorithm.
 * 
 * modified ideas of http://www.hicomb.org/papers/HICOMB2011-06.pdf
 * This version calculates alpha_t for all t at once.
 *
 * Maikel Nadolski <maikel.nadolski@fu-berlin.de>
 */

#ifndef __DIMS__
#define __DIMS__
#define N ((int)${N})
#define M ((int)${M})
#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) A[(t)*N*M + (i)*M + j]
#endif /* __DIMS__ */

#define matrix_times_vector(result,matrix,vector) {  \
   for (int i = 0; i < N; i++) {                     \
      result[i] = 0.0f;                              \
      for (int j = 0; j < N; j++)                    \
         result[i] += matrix[i][j]*vector[j];        \
   }                                                 \
}

#define matrix_times_matrix(result,A,B) {            \
   for (int i = 0; i < N; i++)                       \
      for (int j = 0; j < N; j++) {                  \
         result[i][j] = 0.0f;                        \
         for (int k = 0; k < N; k++)                 \
            result[i][j] += A[i][k]*B[k][j];         \
      }                                              \
}

#define global_to_local(src, gid, dest, lid) {          \
   for (int i = 0; i < N; i++)                          \
      for (int j = 0; j < N; j++)                       \
         DIM3(dest, lid, i, j) = DIM3(src, gid, i, j);  \
}

#define mem_to_private(src, dest, time) {            \
   for (int i = 0; i < N; i++)                       \
      for (int j = 0; j < N; j++)                    \
         dest[i][j] = DIM3(src, time, i, j);         \
}

#define private_to_mem(src, dest, time) {           \
   for (int i = 0; i < N; i++)                      \
      for (int j = 0; j < N; j++)                   \
         DIM3(dest, time, i, j) = src[i][j];        \
}

/*
 * Create the matrices C_t, such that holds alpha_t = C_t * alpha_{t-1}
 */
kernel void build_matrices (
      global   ${precision} *matrices,
      global   ${precision} *alpha,
      constant ${precision} *A,
      constant ${precision} *B,
      constant ${precision} *pi,
      global  short *ob,
      unsigned long T) 
{
   size_t global_id = get_global_id(0);
   ${precision} _A[N][N];
   ${precision} _B[N][M];
   ${precision} _pi[N];
   for (int i = 0; i < N; i++) {
      _pi[i] = pi[i];
      for (int j = 0; j < N; j++)
         _A[i][j] = DIM2(A, i, j);
      for (int k = 0; k < M; k++)
         _B[i][k]  = DIMM2(B, i, k);
   }

   while (global_id < T) {

      short o_t = ob[global_id];

      if (global_id == 0) {
         ${precision} _alpha_0_i;
         for (int i = 0; i < N; i++) {
            _alpha_0_i = _B[i][o_t] * _pi[i];
            DIM2(alpha, global_id, i) = _alpha_0_i;
         }

      } else {
         ${precision} matrices_ij;

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               matrices_ij = _A[j][i] * _B[i][o_t];
               DIM3(matrices, global_id-1, i, j) = matrices_ij;
            }
      }

      global_id += get_global_size(0);
   }
}

kernel void scan (
      global   ${precision} *grouped_results,
      global   ${precision} *last_results,
      local    ${precision} *scratch,
      unsigned long T)
{
   size_t global_id    = get_global_id(0);
   size_t current_root = 0;
   size_t local_id     = get_local_id(0);
   size_t group_id     = get_group_id(0);   
   ${precision} C_t[N][N];

   while (current_root < T) {

      if (global_id < T)
         global_to_local(last_results, global_id, scratch, local_id);

      for (size_t offset = 1;
                  offset < get_local_size(0);
                  offset <<= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE);
         if (global_id < T && local_id >= offset) {
            /* copy matrices from local memory into private memory */
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++) {
                  C_t[i][j] = 0.0f;
                  for (int k = 0; k < N; k++)
                     C_t[i][j] += DIM3(scratch, local_id, i, k)
                           * DIM3(scratch, local_id - offset, k, j);
               }
         }
         barrier(CLK_LOCAL_MEM_FENCE);

         /* store intermediate result in local memory */
         if (global_id < T && local_id >= offset)
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  DIM3(scratch, local_id, i, j) = C_t[i][j];
      }
      
      /* store final results in global memory */
      if (global_id < T) {

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                  DIM3(last_results, global_id, i, j) 
                        = DIM3(scratch, local_id, i, j);

         /* last member of each work group additionaly writes its last
            result into the group table. this table will be used
            sequentially by everyone. in the collection phase */

         if (get_local_size(0)-1 == local_id) {
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  DIM3(grouped_results, group_id, i, j)
                        = DIM3(scratch, local_id, i, j);
         }
      }  
      /* go to next block ... */
      global_id += get_global_size(0);
      current_root += get_global_size(0);
      group_id += get_num_groups(0);
   }
}


kernel void propagate (
      global ${precision} *last_results,
      global ${precision} *grouped_results,
      unsigned long T)
{
   size_t global_id = get_global_id(0);
   size_t group_id  = get_group_id(0);

   if (group_id == 0) {
      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }

   while (global_id < T) {
      ${precision} C_this[N][N];
      ${precision} C_grouped[N][N];
      ${precision} C_new[N][N];

      for (int i = 0; i < N; i++) 
         for (int j = 0; j < N; j++)
            C_this[i][j] = DIM3(last_results, global_id, i, j);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            C_grouped[i][j] = DIM3(grouped_results, group_id-1, i, j);

      matrix_times_matrix(C_new, C_this, C_grouped);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(last_results, global_id, i, j) = C_new[i][j];

      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }
}

kernel void multiply_with_alpha_0(
      global ${precision} *alpha,
      global ${precision} *matrices,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T-1) {
      ${precision} matrix[N][N];
      ${precision} alpha_0[N];
      ${precision} alpha_t[N];

      for (int i = 0; i < N; i++) {
         alpha_0[i] = DIM2(alpha, 0, i);
         for (int j = 0; j < N; j++)
            matrix[i][j] = DIM3(matrices, global_id, i, j);
      }

      matrix_times_vector(alpha_t, matrix, alpha_0);

      for (int i = 0; i < N; i++)
         DIM2(alpha, global_id+1, i) = alpha_t[i];

      global_id += get_global_size(0);
   }
}