#ifndef __DIMS__
#define __DIMS__
#define N ${N}
#define M ${M}
#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) A[(t)*N*M + (i)*M + j]
#endif /* __DIMS__ */


kernel void
append_identity(
      global ${precision} *C,
      unsigned long T,
      unsigned long reminding)
{
   size_t global_id = get_global_id(0);
   if (global_id < reminding)
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            if (i == j)
               DIM3(C, T+global_id, i, j) = 1.0f;
            else
               DIM3(C, T+global_id, i, j) = 0.0f;
}

/* Create the vector of matrices which we want to do the cumulative sum
 * with.
 */
kernel void
forward_build_matrices (
      global   ${precision} *C, /* matrices */
      global   ${precision} *alpha,
      constant ${precision} *A,
      constant ${precision} *B,
      constant ${precision} *pi,
      global   short *ob,
      unsigned long T) 
{
   size_t global_id = get_global_id(0);
    
   if (global_id == 0) {
      for (int i = 0; i < N; i++)
         DIM2(alpha, 0, i) = DIMM2(B, i, ob[0]) * pi[i];
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            if (i == j)
               DIM3(C, 0, i, j) = 1.0f;
            else
               DIM3(C, global_id, i, j) = 0.0f;
      global_id += get_global_size(0);
   }

   while (global_id < T)
   {
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
               DIM3(C, global_id, i, j) = DIMM2(B, i, ob[global_id]) * DIM2(A, j, i);

      global_id += get_global_size(0);
   }
}


void
down_sweep(local ${precision} *scratch)
{
   size_t local_id = get_local_id(0);
   size_t offset = 1;
   /* perform down sweep */
   for (size_t depth = get_local_size(0);
               depth > 0;
               depth >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < depth) {
         int left  = offset*(2*local_id+2)-1; 
         int right = offset*(2*local_id+1)-1; 
         ${precision} C[N][N] = { 0.0f };

         /* matrix multiplication */
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               for (int k = 0; k < N; k++)
                  C[i][j] += DIM3(scratch, left, i, k) * DIM3(scratch, right, k, j);

         /* copy to local mem */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               DIM3(scratch, left, i, j) = C[i][j];
      }
      offset = offset * 2;
   }
}

void
up_sweep(local ${precision} *scratch)
{
   size_t local_id = get_local_id(0);
   size_t offset = get_local_size(0);
   /* perform up sweep */
   for (size_t depth = 1;
               depth <= get_local_size(0);
               depth <<= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < depth) {
         size_t left  = offset*(2*local_id+2)-1; 
         size_t right = offset*(2*local_id+1)-1; 
         ${precision} tmp[N][N] = { 0.0f };
         /* "swap" matrices and multiply C_r * C_l */
         for (int i = 0; i < N; i++) 
            for (int j = 0; j < N; j++)
               for (int k = 0; k < N; k++)
                  tmp[i][j] += DIM3(scratch, right, i, k)*DIM3(scratch, left, k, j);

         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
               DIM3(scratch, right, i, j) = DIM3(scratch, left, i, j);
               DIM3(scratch, left, i, j)  = tmp[i][j];
            }
      }
      offset = offset / 2;
   }
}

/*
 * THIS FUNCTION ASSUMES 
 *
 * !!!! 2*global_id+1 < T !!!!
 */
void
forward_reduce_block(
      global ${precision} *out,
      global ${precision} *in,
      local  ${precision} *scratch,
      size_t global_id,
      size_t group_id,
      unsigned long T)
{
   size_t local_id   = get_local_id(0);
   size_t local_size = get_local_size(0);  
   
   /* get data from global to local memory */ 
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
         DIM3(scratch, 2*local_id, i, j)   = DIM3(in, 2*global_id, i, j);
         DIM3(scratch, 2*local_id+1, i, j) = DIM3(in, 2*global_id+1, i, j);
      }
   

   down_sweep(scratch);

   /* replace last item with the identity and save group result */
   if (local_id == local_size - 1) {
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            DIM3(out, group_id, i, j) = DIM3(scratch, 2*local_id+1, i, j);

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            if (i == j)
               DIM3(scratch, 2*local_id+1, i, i) = 1.0f;
            else
               DIM3(scratch, 2*local_id+1, i, j) = 0.0f;
   }

   up_sweep(scratch);

   barrier(CLK_LOCAL_MEM_FENCE);
   /* write data to output */
   if (local_id != local_size - 1) {
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            DIM3(in, 2*global_id, i, j)   = DIM3(scratch, 2*local_id+1, i, j);
            DIM3(in, 2*global_id+1, i, j) = DIM3(scratch, 2*local_id+2, i, j);
         }
   } else {
   /* write back last entry */
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            DIM3(in, 2*global_id, i, j)   = DIM3(scratch, 2*local_id+1, i, j);
            DIM3(in, 2*global_id+1, i, j) = DIM3(out, group_id, i, j);
         }
   }
}

kernel
void forward_reduce(
      global ${precision} *out,
      global ${precision} *in,
      local  ${precision} *scratch,
      unsigned long T)
{
   size_t global_size = get_global_size(0);
   size_t group_id    = get_group_id(0);
   size_t num_groups  = get_num_groups(0);
   size_t global_id   = get_global_id(0);

   while (2*global_id+1 < T) {

      /* This will crash your Computer if you dont assure, that a hole
         work group joins that function. We do this but padding the `in`
         vector such that T is a multiple of work_group_size. This can
         be done with append_identity called from the host */
      forward_reduce_block(out, in, scratch, global_id, group_id, T);

      global_id += global_size;
      group_id  += num_groups;
   }
}

kernel void
forward_collect (
      global ${precision} *extended,
      global ${precision} *reduced,
      unsigned long T)
{
   size_t global_id = get_global_id(0);
   size_t group_id  = get_group_id(0);

   if (group_id == 0) {
      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }

   while (2*global_id+1 < T) {
      ${precision} C1[N][N] = { 0.0f };
      ${precision} C2[N][N] = { 0.0f };
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
               C1[i][j] += DIM3(extended, 2*global_id, i, k)*DIM3(reduced, group_id-1, k, j);
               C2[i][j] += DIM3(extended, 2*global_id+1, i, k)*DIM3(reduced, group_id-1, k, j);
            }

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
            DIM3(extended, 2*global_id, i, j)   = C1[i][j];
            DIM3(extended, 2*global_id+1, i, j) = C2[i][j];
         }

      global_id += get_global_size(0);
      group_id  += get_num_groups(0);
   }
}

kernel void
forward_build_alpha(
      global ${precision} *alpha,
      global ${precision} *in,
      unsigned long T)
{
   size_t global_id = get_global_id(0);

   while (global_id < T) {
      ${precision} matrix[N][N];
      ${precision} alpha_0[N];
      ${precision} alpha_t[N] = { 0.0f };

      for (int i = 0; i < N; i++) {
         alpha_0[i] = DIM2(alpha, 0, i);
         for (int j = 0; j < N; j++)
            matrix[i][j] = DIM3(in, global_id, i, j);
      }

      /* matrix times vector */
      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
            alpha_t[i] += matrix[i][j]*alpha_0[j];
 
      for (int i = 0; i < N; i++)
         DIM2(alpha, global_id, i) = alpha_t[i];

      global_id += get_global_size(0);
   }
}