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

#define MATRIX_MULTIPLY_NS(AB, A, B) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) { \
      AB[_i][_j] = 0.0f; \
      for (int _k = 0; _k < N; _k++) \
         AB[_i][_j] += A[_i][_k]*B[_k][_j]; \
   }

#define FROM_TO(src, src_id, dest, dest_id) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      DIM3(dest, dest_id, _i, _j) = DIM3(src, src_id, _i, _j)

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

#define COPY_ID(dest) \
for (int _i = 0; _i < N; _i++) \
   for (int _j = 0; _j < N; _j++) \
      if (_i == _j) \
         dest[_i][_j] = 1.0f; \
      else \
         dest[_i][_j] = 0.0f


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
   ${precision} C[N][N];
   ${precision} L[N][N], R[N][N];
   GET_DIM2(C, in, start);
   for (size_t i = start+1; i < end; i++) {
      GET_DIM2(R, in, i);
      MATRIX_MULTIPLY(L, R, C);
      COPY(C, L);
   }
   PUT_DIM2(C, out, out_id);
}

kernel void reduce(
      global ${precision} *in,
      global ${precision} *out,
      local  ${precision} *scratch,
      constant int *transform,
      int T)
{
   size_t NUM_BLOCK_THREADS = get_local_size(0);
   size_t BLOCK_SIZE = NUM_BLOCK_THREADS << 1;
   size_t global_id = get_global_id(0);
   size_t TOTAL_SIZE = BLOCK_SIZE*get_num_groups(0);
   size_t num_corrections = (T / BLOCK_SIZE) % get_num_groups(0);
   size_t sequential_len = T / TOTAL_SIZE;
   size_t T_start;
   if (get_group_id(0) >= num_corrections)
      T_start = (sequential_len+1)*num_corrections*BLOCK_SIZE
            + sequential_len*(2*global_id - num_corrections*BLOCK_SIZE);
   else {
      sequential_len = sequential_len + 1;
      T_start = sequential_len*2*global_id;
   }

   ${precision} C[N][N];
   ${precision} L[N][N];
   ${precision} R[N][N];

   size_t local_id = get_local_id(0);
   size_t left  = transform[2*local_id];
   size_t right = transform[2*local_id+1];

   sum_sequential(scratch, left,
      in, T_start, T_start + sequential_len, T);
   sum_sequential(scratch, right,
      in, T_start + sequential_len, T_start + 2*sequential_len, T);

   barrier(CLK_LOCAL_MEM_FENCE);

   size_t bi = local_id + NUM_BLOCK_THREADS;
   GET_DIM2(C, scratch, bi);
   for (size_t offset = NUM_BLOCK_THREADS;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);

      if (NUM_BLOCK_THREADS <= local_id + offset) {
         size_t ai = bi - offset;
         GET_DIM2(L, scratch, ai);
         MATRIX_MULTIPLY(R, C, L);
         PUT_DIM2(R, scratch, bi);
         COPY(C,R);
      }
   }

   if (local_id == NUM_BLOCK_THREADS-1) 
      PUT_DIM2(C, out, get_group_id(0));
}

void scan(
      local ${precision} *scratch,
      local ${precision} last_element[N][N]
   )
{
   size_t NUM_BLOCK_THREADS = get_local_size(0);
   size_t local_id = get_local_id(0);
   ${precision} C[N][N];
   ${precision} L[N][N] = {0.0f}, R[N][N] = {0.0f};

   size_t bi = local_id + NUM_BLOCK_THREADS;
   GET_DIM2(C, scratch, bi);
   for (size_t offset = NUM_BLOCK_THREADS;
               offset > 0;
               offset >>= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (NUM_BLOCK_THREADS <= local_id + offset) {
         size_t ai = bi - offset;
         GET_DIM2(L, scratch, ai);
         MATRIX_MULTIPLY(R, C, L);
         PUT_DIM2(R, scratch, bi);
         COPY(C,R);
      }  
   }

   if (local_id == NUM_BLOCK_THREADS-1) {
      COPY(last_element, C);
      COPY_ID(C);
      PUT_DIM2(C, scratch, bi);
   }

   for (size_t offset = 1;
               offset <= NUM_BLOCK_THREADS;
               offset <<= 1)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (NUM_BLOCK_THREADS <= local_id + offset) {
         size_t ai = bi - offset;
         GET_DIM2(L, scratch, ai);
         GET_DIM2(C, scratch, bi);
         PUT_DIM2(C, scratch, ai);
         MATRIX_MULTIPLY(R, L, C);
         PUT_DIM2(R, scratch, bi);
      }
   }
}

/*
 * inplace toplevel scan with permutation
 */
kernel void scan_toplevel(
      global ${precision} *in,
      local ${precision} *scratch,
      constant int *transform,
      int T)
{
   size_t NUM_BLOCK_THREADS = get_local_size(0);
//   size_t BLOCK_SIZE = NUM_BLOCK_THREADS << 1;

   size_t local_id = get_local_id(0);
   size_t left  = transform[2*local_id];
   size_t right = transform[2*local_id+1];
   size_t global_id = get_global_id(0);

   local ${precision} last_element[N][N];
   // ${precision} C[N][N];

   if (2*global_id < T) {
      // GET_DIM2(C, in, 2*global_id);
      // PUT_DIM2(C, scratch, left);
      FROM_TO(in, 2*global_id, scratch, left);
   } else
      PUT_ID(scratch, left);

   if (2*global_id+1 < T) {
      // GET_DIM2(C, in, 2*global_id+1);
      // PUT_DIM2(C, scratch, right);
      FROM_TO(in, 2*global_id+1, scratch, right); 
   } else
      PUT_ID(scratch, right);

   barrier(CLK_LOCAL_MEM_FENCE);
   scan(scratch, last_element);
   barrier(CLK_LOCAL_MEM_FENCE);

   if (2*global_id < T) {
      // GET_DIM2(C, scratch, right);
      // PUT_DIM2(C, in, 2*global_id);
      FROM_TO(scratch, right, in, 2*global_id);
   } 

   if (local_id == NUM_BLOCK_THREADS-1 && 2*global_id+1 < T)
      PUT_DIM2(last_element, in, 2*global_id+1);
   if (local_id != 0 && 2*global_id-1 < T) {
      // GET_DIM2(C, scratch, left);
      // PUT_DIM2(C, in, 2*global_id-1);
      FROM_TO(scratch, left, in, 2*global_id-1);
   }
}

/*
 * inplace toplevel scan with permutation
 */
kernel void scan_all(
      global ${precision} *in,
      global ${precision} *toplevel,
      local ${precision} *scratch,
      constant int *transform,
      global ${precision} *debug,
      int T)
{
   size_t NUM_BLOCK_THREADS = get_local_size(0);
   size_t BLOCK_SIZE = NUM_BLOCK_THREADS << 1;
   size_t global_id = get_global_id(0);
   size_t local_id = get_local_id(0);
   size_t TOTAL_SIZE = BLOCK_SIZE*get_num_groups(0);
   size_t num_corrections = (T / BLOCK_SIZE) % get_num_groups(0);
   int sequential_len = T / TOTAL_SIZE;
   size_t T_start;
   if (get_group_id(0) >= num_corrections)
      T_start = (sequential_len+1)*num_corrections*BLOCK_SIZE
            + sequential_len*(2*global_id - num_corrections*BLOCK_SIZE);
   else {
      sequential_len = sequential_len + 1;
      T_start = sequential_len*2*global_id;
   }

   local ${precision} scanned[N][N];
   ${precision} start[N][N];
   ${precision} product[N][N];

   size_t left  = transform[2*local_id];
   size_t right = transform[2*local_id+1];

   if (get_group_id(0) > 0 && local_id == 0) {
      GET_DIM2(scanned, toplevel, get_group_id(0)-1);
      GET_DIM2(start, in, T_start);
      MATRIX_MULTIPLY(product, scanned, start);
      PUT_DIM2(product, scratch, left);
      FROM_TO(in, T_start+1, scratch, right);
   } else {
      FROM_TO(in, T_start+2*local_id,   scratch, left);
      FROM_TO(in, T_start+2*local_id+1, scratch, right);
   }

   FROM_TO(scratch, left,  debug, 2*global_id);
   FROM_TO(scratch, right, debug, 2*global_id+1);

//   while (sequential_len > 0) {
      barrier(CLK_LOCAL_MEM_FENCE);
      scan(scratch, scanned);
      barrier(CLK_LOCAL_MEM_FENCE);

      // FROM_TO(scratch, left,  debug, 2*global_id);
      // FROM_TO(scratch, right, debug, 2*global_id+1);

      if (local_id == NUM_BLOCK_THREADS-1)
         PUT_DIM2(scanned, in, T_start+2*local_id+1);
      
      FROM_TO(scratch, right, in, T_start+2*local_id);
      if (get_local_id(0) != 0)
         FROM_TO(scratch, left, in, T_start+2*local_id-1);

      T_start += BLOCK_SIZE;
      sequential_len -= 1;

      // if (local_id == 0) {
      //    GET_DIM2(start, in, T_start);
      //    MATRIX_MULTIPLY(product, scanned, start);
      //    PUT_DIM2(product, scratch, left);
      //    FROM_TO(in, T_start+1, scratch, right);
      // } else {
      //    FROM_TO(in, T_start+2*local_id,   scratch, left);
      //    FROM_TO(in, T_start+2*local_id+1, scratch, right);
      // }
  // }
   //   sequential_len -= 1;
   //   T_start += BLOCK_SIZE;
      // if (get_local_id(0) == 0) {
      //    GET_DIM2(A, in, T_start-1);
      //    GET_DIM2(C_0, in, T_start);
      //    MATRIX_MULTIPLY_NS(C_0A, C_0, A);
      //    PUT_DIM2(C_0A, in, T_start);
      // }
   //}
   // scan(in + T_start, scratch, transform, debug, BLOCK_SIZE);
}