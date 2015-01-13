#ifndef __DIMS__
#define __DIMS__

#define N ${N}
#define M ${M}

#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) B[(t)*N*M + (i)*M + j]

#define GET_DIM2(dest, src, src_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      dest[_i][_j] = DIM3(src, src_id, _i, _j)

#define GET_DIMM2(dest, src, src_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _k = 0; _k < M; _k++) \
      dest[_i][_k] = DIMM3(src, src_id, _i, _k)

#define PUT_DIM2(src, dest, dest_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      DIM3(dest, dest_id, _i, _j) = src[_i][_j]

#define PUT_DIM3(src, src_id, dest, dest_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      DIM3(dest, dest_id, _i, _j) = DIM3(src, src_id, _i, _j)

#define PUT_ID(dest, dest_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      if (_i == _j) \
         DIM3(dest, dest_id, _i, _j) = 1.0f; \
      else \
         DIM3(dest, dest_id, _i, _j) = 0.0f

#endif /* __DIMS__ */

#define MATRIX_MULTIPLY(AB, A, B) \
${precision} __sum = 0.0f; \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) { \
      AB[_i][_j] = 0.0f; \
      for (int _k = 0; _k < N; _k++) \
         AB[_i][_j] += A[_i][_k]*B[_k][_j]; \
      __sum += AB[_i][_j]; \
   } \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      AB[_i][_j] /= __sum

#define COPY(dest, src) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      dest[_i][_j] = src[_i][_j];


/* Create the vector of matrices which we want to do the cumulative sum
 * with.
 */
kernel void build_matrices (
      global   ${precision} *C, /* matrices */
      constant ${precision} *A,
      constant ${precision} *B,
      constant ${precision} *pi,
      global   short *seq,
      int T) 
{
   size_t global_id = get_global_id(0);
   size_t global_size = get_global_size(0);
   
   ${precision} _A[N][N];
   ${precision} _B[N][M];
   GET_DIM2(_A, A, 0);
   GET_DIMM2(_B, B, 0);

   if (global_id == 0) {
      PUT_ID(C, 0);
      global_id += global_size;
   }

   while (global_id < T)
   {
      short ob = seq[global_id];
      ${precision} _C[N][N];

      for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
               _C[i][j] = _B[i][ob]*_A[j][i];

      PUT_DIM2(_C, C, global_id);

      global_id += global_size;
   }
}


void sum_sequential(
      local  ${precision} *out,
      size_t out_id,
      global ${precision} *in,
      size_t start,
      size_t end,
      int T)
{
   if (start < T) {
      ${precision} C[N][N];
      ${precision} L[N][N], R[N][N];
      GET_DIM2(C, in, start);
      for (size_t i = start+1; i < end && i < T; i++) {
         GET_DIM2(R, in, i);
         MATRIX_MULTIPLY(L, R, C);
         COPY(C, L);
      }
      PUT_DIM2(C, out, out_id);
   } else 
      PUT_ID(out, out_id);
}

kernel void reduce(
      global ${precision} *in,
      global ${precision} *out,
      local  ${precision} *scratch,
      constant int *transform,
      int T)
{
   size_t HALF_BLOCK = get_local_size(0);
   size_t BLOCK_SIZE = HALF_BLOCK << 1;

   size_t T_seq = T / (BLOCK_SIZE * get_num_groups(0)) + 1;

   ${precision} C[N][N];
   ${precision} L[N][N] = {0.0f}, R[N][N] = {0.0f};

   size_t local_id = get_local_id(0);
   size_t left  = transform[2*local_id];
   size_t right = transform[2*local_id+1];
   size_t gid = get_global_id(0);

   sum_sequential(scratch, left,
      in, 2*gid*T_seq, (2*gid+1)*T_seq, T);
   sum_sequential(scratch, right,
      in, (2*gid+1)*T_seq, (2*gid+2)*T_seq, T);

   barrier(CLK_LOCAL_MEM_FENCE);

   size_t bi = local_id + HALF_BLOCK;
   GET_DIM2(C, scratch, bi);
   for (size_t offset = HALF_BLOCK;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (HALF_BLOCK <= local_id + offset) {
         size_t ai = bi - offset;
         GET_DIM2(L, scratch, ai);
         MATRIX_MULTIPLY(R, C, L);
         PUT_DIM2(R, scratch, bi);
         COPY(C,R);
      }
   }

   if (local_id == HALF_BLOCK-1)
      PUT_DIM2(C, out, get_group_id(0));
}


kernel void scan(
      global ${precision} *in,
      global ${precision} *out,
      local ${precision} *scratch,
      constant int *transform,
      int T)
{
   size_t HALF_BLOCK = get_local_size(0);
   size_t BLOCK_SIZE = HALF_BLOCK << 1;

   size_t local_id = get_local_id(0);
   size_t left  = transform[2*local_id];
   size_t right = transform[2*local_id+1];
   size_t global_id = get_global_id(0);
   size_t group_root = global_id - local_id;

   size_t T_segment = T / (get_num_groups(0)*BLOCK_SIZE)

   local ${precision} last_element[N][N];
   ${precision} C[N][N];
   ${precision} L[N][N] = {0.0f}, R[N][N] = {0.0f};

   while (group_root < T_segment) {

      GET_DIM2(C, in, 2*global_id);
      PUT_DIM2(C, scratch, left);
      GET_DIM2(C, in, 2*global_id+1);
      PUT_DIM2(C, scratch, right);

      barrier(CLK_LOCAL_MEM_FENCE);
      size_t bi = local_id + HALF_BLOCK;
      GET_DIM2(C, scratch, bi);
      for (size_t offset = HALF_BLOCK;
                  offset > 0;
                  offset >>= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE);
         if (HALF_BLOCK <= local_id + offset) {
            size_t ai = bi - offset;
            GET_DIM2(L, scratch, ai);
            MATRIX_MULTIPLY(R, C, L);
            PUT_DIM2(R, scratch, bi);
            COPY(C,R);
         }  
      }

      if (local_id == HALF_BLOCK-1) {
         COPY(last_element, C);
         PUT_ID(*C, 0);
      }

      for (size_t offset = 1;
                  offset <= HALF_BLOCK;
                  offset <<= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE);
         if (HALF_BLOCK <= local_id + offset) {
            size_t ai = bi - offset;
            GET_DIM2(L, scratch, ai);
            MATRIX_MULTIPLY(R, L, C);
            PUT_DIM2(C, scratch, ai);
            COPY(C,R);
         }
      }

      PUT_DIM2(C, in, 2*global_id);
      if (local_id == HALF_BLOCK-1)
         PUT_DIM2(last_element, in, 2*global_id+1);
      else if (local_id != 0) {
         GET_DIM2(C, scratch, local_id);
         PUT_DIM2(C, in, 2*global_id-1);
      }

   }
}