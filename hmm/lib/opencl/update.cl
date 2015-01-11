#ifndef __DIMS__
#define __DIMS__
#define N ((int)${N})
#define M ((int)${M})
#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) B[(t)*N*M + (i)*M + j]
#endif /* __DIMS__ */

kernel void
transition_probabilities(
      global ${precision} *transition_probs,
      global ${precision} *alpha,
      global ${precision} *beta,
      constant ${precision} *A,
      constant ${precision} *B,
      global short *ob,
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
         _B[i][k] = DIMM2(B, i, k);

   while (global_id+1 < T) {
      ${precision} alpha_t[N];
      for (int i = 0; i < N; i++)
         alpha_t[i] = DIM2(alpha, global_id, i);

      ${precision} beta_t1[N];
      for (int i = 0; i < N; i++)
         beta_t1[i] = DIM2(beta, global_id+1, i);

      short o_t1 = ob[global_id+1];

      ${precision} xi_t[N][N];
      ${precision} sum = 0.0f;
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            xi_t[i][j] = alpha_t[i]*_A[i][j]*_B[j][o_t1]*beta_t1[j];
            sum += xi_t[i][j];
         }

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            xi_t[i][j] = xi_t[i][j] / sum;
            DIM3(transition_probs, global_id, i, j) = xi_t[i][j];
         }

      global_id += get_global_size(0);
   }
}

kernel void
transition_counts(
      global ${precision} *transition_probs,
      local  ${precision} *scratch,
      unsigned long T,
      global ${precision} *transition_cts)
{
   size_t global_id = get_global_id(0);
   size_t global_size = get_global_size(0);
   ${precision} xi[N][N] = { 0.0f };

   while (global_id < T) {
      ${precision} tmp[N][N];
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            tmp[i][j] = DIM3(transition_probs, global_id, i, j);
            xi[i][j] += tmp[i][j];
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
               offset >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < offset) {
         ${precision} other[N][N];

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               other[i][j] = DIM3(scratch, local_id+offset, i, j);
               xi[i][j] += other[i][j];
               DIM3(scratch, local_id, i, j) = xi[i][j];
            }
      }
   }

   size_t group_id = get_group_id(0);
   if (local_id == 0)
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(transition_cts, group_id, i, j) = xi[i][j];
}

kernel void
state_probabilities(
      global ${precision} *alpha,
      global ${precision} *beta,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T) {
      ${precision} alpha_t[N], beta_t[N];
      ${precision} sum = 0.0f;
      for (int i = 0; i < N; i++) {
         alpha_t[i] = DIM2(alpha, global_id, i);
         beta_t[i]  = DIM2(beta, global_id, i);
      }
      for (int i = 0; i < N; i++) {
         alpha_t[i] *= beta_t[i];
         sum += alpha_t[i];
      }
      for (int i = 0; i < N; i++)
         alpha_t[i] /= sum;

      for (int i = 0; i < N; i++)
         DIM2(alpha, global_id, i) = alpha_t[i];


      global_id += get_global_size(0);
   }
}

kernel void
state_counts(
      global ${precision} *gamma,  
      local  ${precision} *scratch,
      unsigned long T,
      global ${precision} *gamma_counts)
{
   size_t global_id = get_global_id(0);
   ${precision} gamma_t[N] = { 0.0f };

   while (global_id < T) {
      for (int i = 0; i < N; i++) {
         gamma_t[i] += DIM2(gamma, global_id, i);
      }
      global_id += get_global_size(0);
   }

   size_t local_id = get_local_id(0);
   for (int i = 0; i < N; i++)
      DIM2(scratch, local_id, i) = gamma_t[i];
   for (size_t offset = get_local_size(0) >> 1;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < offset)
         for (int i = 0; i < N; i++) {
            gamma_t[i] += DIM2(scratch, local_id+offset, i);
            DIM2(scratch, local_id, i) = gamma_t[i];
         }
   }

   if (local_id == 0)
      for (int i = 0; i < N; i++)
         DIM2(gamma_counts, get_group_id(0), i) = gamma_t[i];
}

kernel void
symbol_counts(
      global ${precision} *gamma,
      global short *ob,
      local ${precision} *scratch,
      unsigned long T,
      global ${precision} *symbols
      )
{
   size_t global_id = get_global_id(0);
   ${precision} B[N][M] = { 0.0f };

   while (global_id < T) {
      short o_t = ob[global_id];
      for (int i = 0; i < N; i++)
            B[i][o_t] += DIM2(gamma, global_id, i);
      global_id += get_global_size(0);
   }

   size_t local_id = get_local_id(0);
   for (int i = 0; i < N; i++)
      for (int k = 0; k < M; k++)
         DIMM3(scratch, local_id, i, k) = B[i][k];
   for (size_t offset = get_local_size(0) >> 1;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < offset)
         for (int i = 0; i < N; i++)
            for (int k = 0; k < M; k++) {
               B[i][k] += DIMM3(scratch, local_id+offset, i, k);
               DIMM3(scratch, local_id, i, k) = B[i][k];
            }
   }

   if (local_id == 0)
      for (int i = 0; i < N; i++)
         for (int k = 0; k < M; k++)
            DIMM3(symbols, get_group_id(0), i, k) = B[i][k];//DIM3(scratch, local_id, i, k);
}

kernel void
symbol_collect(
      global ${precision} *symbols_intermediate,
      global short *ob,
      local ${precision} *scratch,
      unsigned long T,
      global ${precision} *symbols
      )
{
   size_t global_id = get_global_id(0);
   ${precision} B[N][M] = { 0.0f };

   while (global_id < T) {
      for (int i = 0; i < N; i++)
         for (int k = 0; k < M; k++)
            B[i][k] += DIMM3(symbols_intermediate, global_id, i, k);
      global_id += get_global_size(0);
   }

   size_t local_id = get_local_id(0);
   for (int i = 0; i < N; i++)
      for (int k = 0; k < M; k++)
         DIMM3(scratch, local_id, i, k) = B[i][k];
   for (size_t offset = get_local_size(0) >> 1;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < offset)
         for (int i = 0; i < N; i++)
            for (int k = 0; k < M; k++) {
               B[i][k] += DIMM3(scratch, local_id+offset, i, k);
               DIMM3(scratch, local_id, i, k) = B[i][k];
            }
   }

   if (local_id == 0)
      for (int i = 0; i < N; i++)
         for (int k = 0; k < M; k++)
            DIMM3(symbols, get_group_id(0), i, k) = B[i][k];//DIM3(scratch, local_id, i, k);
}

kernel void
update(
      global ${precision} *A,
      global ${precision} *B,
      global ${precision} *pi,
      global ${precision} *gamma,
      global ${precision} *transition_counts,
      global ${precision} *state_counts,
      global ${precision} *symbol_counts,
      global ${precision} *probability,
      unsigned long T)
{
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);

   if (i < N && j < N) {
      if (j == 0)
         pi[i] = DIM2(gamma, 0, i);

      DIM2(A, i, j) = DIM2(transition_counts, i, j) / state_counts[i];

      if (j < M)
         DIMM2(B, i, j) = DIMM2(symbol_counts, i, j) / (state_counts[i] + DIM2(gamma, T-1, i));
   }
}