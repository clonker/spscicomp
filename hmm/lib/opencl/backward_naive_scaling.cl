/*
 * parallel GPU optimized kernel implementation of the backward algorithm.
 * 
 * modified ideas of http://www.hicomb.org/papers/HICOMB2011-06.pdf
 * This version calculates beta_t for all t at once.
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

#ifndef __MULTIPLICATIONS__
#define __MULTIPLICATIONS__
#define matrix_times_vector(result,matrix,vector) {  \
   ${precision} scaling = 0.0f;                      \
   for (int i = 0; i < N; i++) {                     \
      result[i] = 0.0f;                              \
      for (int j = 0; j < N; j++)                    \
         result[i] += matrix[i][j]*vector[j];        \
      scaling += result[i];                          \
   }                                                 \
   if (scaling != 0) for (int i = 0; i < N; i++)     \
      result[i] /= scaling;                          \
}

#define matrix_times_matrix(result,A,B) {            \
   ${precision} scaling = 0.0f;                      \
   for (int i = 0; i < N; i++)                       \
      for (int j = 0; j < N; j++) {                  \
         result[i][j] = 0.0f;                        \
         for (int k = 0; k < N; k++)                 \
            result[i][j] += A[i][k]*B[k][j];         \
         scaling += result[i][j];                    \
      }                                              \
   if (scaling != 0) for (int i = 0; i < N; i++)     \
      for (int j = 0; j < N; j++)                    \
         result[i][j] /= scaling;                    \
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
#endif

/*
 * Create the matrices C_t, such that holds beta_t = C_t * beta_{t+1}
 */
kernel void build_matrices (
      global   ${precision} *matrices,
      global   ${precision} *beta,
      constant ${precision} *A,
      constant ${precision} *B,
      global  short *ob,
      unsigned long T) 
{
   size_t global_id = get_global_id(0);
   ${precision} _A[N][N];

   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         _A[i][j] = DIM2(A, i, j);

   /* copy data to private memory first */
   ${precision} _B[N][M];
   for (int i = 0; i < N; i++)
      for (int k = 0; k < M; k++)
         _B[i][k]  = DIMM2(B, i, k);

   while (global_id < T) {

      short o_t = ob[global_id];

      if (global_id == 0) {
         for (int i = 0; i < N; i++)
            DIM2(beta, T-1, i) = 1.0f / N;

      } else {
         ${precision} matrices_ij;

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               matrices_ij = _A[i][j] * _B[j][o_t];
               DIM3(matrices, global_id-1 , i, j) = matrices_ij;
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
   size_t local_id     = get_local_id(0);
   size_t current_root = global_id - local_id;
   size_t group_id     = get_group_id(0); 
   size_t local_size   = get_local_size(0);  
   ${precision} C_t[N][N];

   if (local_size > T) {
      local_size = T;
   }

   while (current_root < T) {

      if (global_id >= T)
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               if (i == j)
                  DIM3(scratch, local_id, i, j) = 1.0f;
               else
                  DIM3(scratch, local_id, i, j) = 0.0f;

      if (global_id < T)
         global_to_local(last_results, global_id, scratch, local_id);

      for (size_t offset = 1;
                  offset < local_size;
                  offset <<= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE);
         if (global_id < T && local_id+offset < local_size) {
            ${precision} scaling_factor = 0.0f;
            /* copy matrices from local memory into private memory */
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++) {
                  C_t[i][j] = 0.0f;
                  for (int k = 0; k < N; k++)
                     C_t[i][j] += DIM3(scratch, local_id, i, k) * DIM3(scratch, local_id + offset, k, j);
                  scaling_factor += C_t[i][j];
               }
            /* rescale C_t */
            if (scaling_factor != 0)
               for (int i = 0; i < N; i++)
                  for (int j = 0; j < N; j++)
                     C_t[i][j] /= scaling_factor;
         }
         barrier(CLK_LOCAL_MEM_FENCE);

         /* store intermediate result in local memory */
         if (global_id < T && local_id+offset < local_size)
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

         if (0 == local_id) {
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
   size_t num_groups = T / get_local_size(0);
   if (T % get_local_size(0) != 0)
      num_groups += 1;

   while (global_id < T && group_id+1 < num_groups) {
      ${precision} C_this[N][N];
      ${precision} C_grouped[N][N];
      ${precision} C_new[N][N];

      for (int i = 0; i < N; i++) 
         for (int j = 0; j < N; j++)
            C_this[i][j] = DIM3(last_results, global_id, i, j);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            C_grouped[i][j] = DIM3(grouped_results, group_id+1, i, j);

      matrix_times_matrix(C_new, C_this, C_grouped);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(last_results, global_id, i, j) = C_new[i][j];

      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }
}

kernel void multiply_with_beta_T(
      global ${precision} *beta,
      global ${precision} *matrices,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T-1) {
      ${precision} matrix[N][N];
      ${precision} beta_T[N];
      ${precision} beta_t[N];

      for (int i = 0; i < N; i++) {
         beta_T[i] = DIM2(beta, T-1, i);
         for (int j = 0; j < N; j++)
            matrix[i][j] = DIM3(matrices, global_id, i, j);
      }

      matrix_times_vector(beta_t, matrix, beta_T);

      for (int i = 0; i < N; i++)
         DIM2(beta, global_id, i) = beta_t[i];

      global_id += get_global_size(0);
   }
}