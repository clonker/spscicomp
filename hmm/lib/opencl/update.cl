#ifndef __DIMS__
#define __DIMS__
#define N ((int)${N})
#define M ((int)${M})
#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) A[(t)*N*M + (i)*M + j]
#endif /* __DIMS__ */

kernel void
transition_probabilities(
      global ${precision} *transition_probs,
      global ${precision} *alpha,
      global ${precision} *beta,
      constant ${precision} *A,
      constant ${precision} *B,
      global ${precision} *ob,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   ${precision} _A[N][N];
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         _A[i][j] = DIM2(A, i, j);

   ${precision} _B[N][M];
   for (int i = 0; i < N; i++)
      for (int k = 0; k < M; k++)
         _B[i][k] = DIM2(B, i, k);

   while (global_id+1 < T) {
      ${precision} alpha_t[N];
      for (int i = 0; i < N; i++)
         alpha_t[i] = DIM2(alpha, global_id, i);

      ${precision} beta_t[N];
      for (int i = 0; i < N; i++)
         beta_t1[i] = DIM2(beta, global_id+1, i);

      ${precision} o_t1 = ob[global_id+1]

      ${precision} xi_t[N][N];
      ${precision} sum;
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            xi_t[i][j] = alpha_t[i]*_A[i][j]*B[j][o_t1]*beta_t1[j];
            sum += xi_t[i][j];
         }

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(transition_probs, global_id, i, j) = xi_t[i][j] / sum;

      global_id += get_global_size(0);
   }
}

kernel void
transition_counts(
      global ${precision} *transition_probs,
      local  ${precision} *scratch,
      int T,
      global ${precision} *transition_counts)
{
   size_t global_id = get_global_id(0);
   size_t global_size = get_global_size(0);
   ${precision} xi[N][N] = { 0.0f; };
   while (global_id < T) {
      ${precision} tmp[N][N];
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            tmp[i][j] = DIM3(transition_probs, global_id, i, j);
            x[i][j] += tmp[i][j];
         }
      global_id += global_size;
   }

   // perform parallel reduction
   size_t local_id = get_local_id(0);
   size_t local_size = get_local_size(0);
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         DIM3(scratch, local_id, i, j) = xi[i][j];
   for (size_t offset = local_size >> 1;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_index < offset) {
         ${precision} other[N][N];

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               other[i][j] = DIM3(scratch, local_id+offset, i, j);
               xi[i][j] += other[i][j];
            }
      }
   }
   size_t group_id = get_group_id(0);
   if (local_id == 0)
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(transition_counts, group_id, i, j) = xi[i][j];
}