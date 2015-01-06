#ifndef __DIMS__
#define __DIMS__
#define N ${N}
#define M ${M}
#define DIM2(A,i,j) A[(i)*N+j]
#define DIMM2(B,i,k) B[(i)*M+k]
#define DIM3(A,t,i,j) A[(t)*N*N + (i)*N + j]
#define DIMM3(B,t,i,j) A[(t)*N*M + (i)*M + j]
#endif /* __DIMS__ */

/*  Create the matrices which we want to do the scan algorithm with. */
kernel void
forward_build_matrices (
      global   ${precision} *C, /* matrices */
      global   ${precision} *alpha,
      constant ${precision} *A,
      constant ${precision} *B,
      constant ${precision} *pi,
      global   short *ob,
      uint T) 
{
   size_t t = get_global_id(0);
    
   while (t < T)
   {
      if (t == 0)
         for (int i = 0; i < N; i++)
            DIM2(alpha, 0, i) = DIM2(B, i, ob[0]) * pi[i];

      else
         for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
               DIM3(C, t-1, i, j) = DIM2(B, i, ob[t]) * DIM2(A, j, i);

      global_id += get_global_size(0);
   }
}


kernel void
forward_reduction(
      global ${precision} *out,
      global ${precision} *in,
      local  ${precision} *temp,
      uint T)
{
   size_t t = get_global_id(0);

   while (t < T)
   {

   }
}