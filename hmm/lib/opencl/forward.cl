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

#define global_to_local(src, gid, dest, lid) {         \
   for (int i = 0; i < N; i++)                         \
      for (int j = 0; j < N; j++)                      \
         DIM3(dest, lid, i, j) = DIM3(src, gid, i, j);  \
}

#define mem_to_private(src, dest, time) {           \
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
kernel void
forward_no_scaling_build_matrices (
      global   float *matrices,
      global   float *alpha,
      constant float *A,
      constant float *B,
      constant float *pi,
      global  short *ob,
      unsigned long T) 
{
   size_t global_id = get_global_id(0);
    
   while (global_id < T) {

      /* copy data to private memory first */
      float _B[N];
      float _pi[N];
      for (int i = 0; i < N; i++) {
         _B[i]  = DIMM2(B, i, ob[global_id]);
         _pi[i] = pi[i];
      }


      if (global_id == 0) {
         float _alpha_0[N];

         for (int i = 0; i < N; i++) {
            _alpha_0[i] = _B[i] * _pi[i];
            DIM2(alpha, global_id, i) = _alpha_0[i];
         }

      } else {
         float _A[N][N], _matrices[N][N];
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               _A[i][j] = DIM2(A, j, i);

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               _matrices[i][j] = _A[i][j] * _B[i];
               DIM3(matrices, global_id-1, i, j) = _matrices[i][j];
            }
      }

      global_id += get_global_size(0);
   }
}

kernel void
forward_build_matrices(
      global float *matrices,
      global float *alpha,
      global float *scaling,
      constant float *A,
      constant float *B,
      constant float *pi,
      global  short *ob,
      unsigned long T
      )
{
   size_t global_id = get_global_id(0);
    
   while (global_id < T) {

      /* copy data to private memory first */
      float _B[N];
      float _pi[N];
      for (int i = 0; i < N; i++) {
         _B[i]  = DIMM2(B, i, ob[global_id]);
         _pi[i] = pi[i];
      }

      scaling[global_id] = 1.0f;
      float sum = 0.0f;

      if (global_id == 0) {
         float _alpha_0[N];
         for (int i = 0; i < N; i++) {
            _alpha_0[i] = _B[i] * _pi[i];
         }
         for (int i = 0; i < N; i++) {
            DIM2(alpha, global_id, i) = _alpha_0[i] / sum;
         }

      } else {
         float _A[N][N], _matrices[N][N];
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               _A[i][j] = DIM2(A, i, j);
         for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
               _matrices[i][j] = _A[i][j] * _B[i];
            }
         }
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               DIM3(matrices, global_id-1, i, j) = _matrices[i][j];
      }

      global_id += get_global_size(0);
   }   
}

kernel void
forward_reduce_naive (
      global   float *grouped_results,
      global   float *last_results,
      global   float *scaling,
      local    float *scratch,
      unsigned long T)
{
   size_t global_id    = get_global_id(0);
   size_t current_root = 0;
   size_t local_id     = get_local_id(0);
   size_t group_id     = get_group_id(0);   
   float C_t[N][N];
   //float sum;

   while (current_root < T) {

      if (global_id < T)
         global_to_local(last_results, global_id, scratch, local_id);

      barrier(CLK_LOCAL_MEM_FENCE);
      for (size_t offset = 1;
                  offset < get_local_size(0);
                  offset <<= 1)
      {
         if (global_id < T && local_id >= offset) {            
            /* copy matrices from local memory into private memory */
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++) {
                  C_t[i][j] = 0.0f;
                  for (int k = 0; k < N; k++)
                     C_t[i][j] += DIM3(scratch, local_id, i, k)*DIM3(scratch, local_id - offset, k, j);
               }
         }
         barrier(CLK_LOCAL_MEM_FENCE);

         /* store intermediate result in local memory */
         if (global_id < T && local_id >= offset)
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  DIM3(scratch, local_id, i, j) = C_t[i][j];

         barrier(CLK_LOCAL_MEM_FENCE);
      }
      
      /* store final results in global memory and perform proper scaling */
      if (global_id < T) {
         float scaling_factor = 0.0f;
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) { 
               C_t[i][j] = DIM3(scratch, local_id, i, j);
               scaling_factor += C_t[i][j];
            }
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N ;j++) {
               DIM3(last_results, global_id, i, j) = C_t[i][j] / scaling_factor;
               scaling[global_id] *= scaling_factor;
            }

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

kernel void
forward_rewind (
      global float *last_results,
      global float *grouped_results,
      global float *scaling,
      unsigned long T)
{
   size_t global_id = get_global_id(0);
   size_t group_id  = get_group_id(0);

   if (group_id == 0) {
      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }

   while (global_id < T) {
      float C_this[N][N];
      float C_grouped[N][N];
      float C_new[N][N] = { 0.0f };

      for (int i = 0; i < N; i++) 
         for (int j = 0; j < N; j++)
            C_this[i][j] = DIM3(last_results, global_id, i, j);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            C_grouped[i][j] = DIM3(grouped_results, group_id-1, i, j);

      float sum = 0.0f;
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
               C_new[i][j] += C_this[i][k]*C_grouped[k][j];
               sum += C_new[i][j];
            }
      scaling[global_id] *= sum;
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            C_new[i][j] /= sum;

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(last_results, global_id, i, j) = C_new[i][j];

      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }
}

kernel void
forward_multiply_with_alpha_0(
      global float *alpha,
      global float *matrices,
      global float *scaling,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T-1) {
      float matrix[N][N];
      float alpha_0[N];
      float alpha_t[N] = { 0.0f };

      for (int i = 0; i < N; i++) {
         alpha_0[i] = DIM2(alpha, 0, i);
         for (int j = 0; j < N; j++)
            matrix[i][j] = DIM3(matrices, global_id, i, j);
      }

      float sum = 0.0f;
      for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
            alpha_t[i] += matrix[i][j]*alpha_0[j];
         }
         sum += alpha_t[i];
      }
      scaling[global_id+1] *= sum;

      for (int i = 0; i < N; i++)
         DIM2(alpha, global_id+1, i) = alpha_t[i] / sum;

      global_id += get_global_size(0);
   }
}






kernel void
forward_no_scaling_reduce_naive (
      global   float *grouped_results,
      global   float *last_results,
      local    float *scratch,
      unsigned long T)
{
   size_t global_id    = get_global_id(0);
   size_t current_root = 0;
   size_t local_id     = get_local_id(0);
   size_t group_id     = get_group_id(0);   
   float C_t[N][N];

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
                      C_t[i][j] += DIM3(scratch, local_id, i, k)*DIM3(scratch, local_id - offset, k, j);
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


kernel void
forward_no_scaling_reduce_belloch(
      global float *grouped_results,
      global float *last_results,
      local float *scratch,
      unsigned long T)
{
   size_t global_id = get_global_id(0);
   size_t current_root = 0;
   size_t local_id  = get_local_id(0);
   size_t group_id = get_group_id(0);
   int offset = 1;

   while (current_root < T) {

      /* load input into local memory */
      if (global_id < T)
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               DIM3(scratch, local_id, i, j) = DIM3(last_results, global_id, i, j);

      for (int d = get_local_size(0) >> 1; d > 0; d >>= 1) {
         barrier(CLK_LOCAL_MEM_FENCE);
         
         if (local_id < d && global_id < T) {
            int ai = offset*(2*local_id+1)-1;
            int bi = offset*(2*local_id+2)-1;

            float C[N][N] = { 0.0f };
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  for (int k = 0; k < N; k++)
                     C[i][j] += DIM3(scratch, ai, i, k)*DIM3(scratch, bi, k, j);
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  DIM3(scratch, bi, i, j) = C[i][j];
         }
         offset <<= 1;
      }
      
      if (local_id == get_local_size(0)-1) {
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               DIM3(grouped_results, group_id, i, j) = DIM3(scratch, local_id, i, j);
               DIM3(scratch, local_id, i, j) = 0.0f;
            }
      }

      for (int d = 1; d < get_local_size(0); d <<= 1) {
         offset >>= 1;
         barrier(CLK_LOCAL_MEM_FENCE);

         if (local_id < d && global_id < T) {
            int ai = offset*(2*local_id+1)-1;
            int bi = offset*(2*local_id+2)-1;

            float C[N][N] = { 0.0f }, tmp[N][N];
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++) {
                  tmp[i][j] = DIM3(scratch, ai, i, j);
                  DIM3(scratch, ai, i, j) = DIM3(scratch, bi, i, j);
               }
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  for (int k = 0; k < N; k++)
                     C[i][j] += tmp[i][j]*DIM3(scratch, ai, i, k);
            for (int i = 0; i < N; i++)
               for (int j = 0; j < N; j++)
                  DIM3(scratch, bi, i, j) = C[i][j];               
         }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (global_id < T)
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(last_results, global_id, i, j) = DIM3(scratch, local_id, i, j);
      global_id += get_global_size(0);
      current_root += get_global_size(0);
      group_id += get_num_groups(0);
   }
}

kernel void
forward_no_scaling_rewind (
      global float *last_results,
      global float *grouped_results,
      unsigned long T)
{
   size_t global_id = get_global_id(0);
   size_t group_id  = get_group_id(0);

   if (group_id == 0) {
      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }

   while (global_id < T) {
      float C_this[N][N];
      float C_grouped[N][N];
      float C_new[N][N];

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

kernel void
forward_no_scaling_multiply_with_alpha_0(
      global float *alpha,
      global float *matrices,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T-1) {
      float matrix[N][N];
      float alpha_0[N];
      float alpha_t[N];

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