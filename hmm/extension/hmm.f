      SUBROUTINE FORWARD(A,B,PI,O,ALPHA,C,PROB,T,N,M)
      INTEGER N, M, T, S, I, J
      REAL*8 A(N,N), B(N,M), PI(N)
      INTEGER O(T)
      REAL*8 ALPHA(T,N), C(T), PROB
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) t
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(in) pi
Cf2py intent(in) o
Cf2py intent(out) alpha
Cf2py intent(out) c
Cf2py intent(out) prob

c  initial values
      DO I = 1, N
      	ALPHA(1,I)=PI(I)*B(I,O(1)+1)
      	C(1) = C(1) + ALPHA(1, I)
      ENDDO
      DO I = 1, N
      	ALPHA(1, I) = ALPHA(1, I) / C(1)
      ENDDO

c  Induction
      DO S = 1, T-1
      	C(S+1) = 0.0d0
      	DO J = 1, N
      		ALPHA(S+1,J)=0.0d0
      		DO I = 1, N
     				ALPHA(S+1,J)=ALPHA(S+1,J)+ALPHA(S,I)*A(I,J)
      		ENDDO
      		ALPHA(S+1, J) = ALPHA(S+1,J)*B(J,O(S+1)+1)
      		C(S+1) = C(S+1) + ALPHA(S+1, J)
      	ENDDO
      	do J = 1, N
      		ALPHA(S+1, J) = ALPHA(S+1, J) / C(S+1)
      	ENDDO
      ENDDO

c  calculate logarithm of probability
      PROB = 0.0d0
      DO S = 1, T
      	PROB = PROB + LOG(C(S))
      ENDDO
      END


      SUBROUTINE BACKWARD(A,B,O,BETA,C,T,N,M)
      INTEGER N, M, T, S, I, J
      REAL*8 A(N,N), B(N,M)
      INTEGER O(T)
      REAL*8 BETA(T,N), C(T)
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) t
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(in) o
Cf2py intent(in) c
Cf2py intent(out) BETA

c  initial values
      DO I = 1, N
      	BETA(T,I)=1.0d0 / C(T)
      ENDDO

c  Induction
      DO S = T-1,1,-1
      	DO I = 1, N
      		BETA(S,I) = 0.0d0
      		DO J = 1, N
      			BETA(S,I)=BETA(S,I)+A(I,J)*BETA(S+1,J)*B(J,O(S+1)+1)
      		ENDDO
      		BETA(S,I) = BETA(S,I)/C(S)
      	ENDDO
      ENDDO
      END


      SUBROUTINE COMPUTEGAMMA(ALPHA,BETA,G,T,N)
      INTEGER T, N, S, I
      REAL*8 ALPHA(T,N), BETA(T,N)
      REAL*8 G(T,N), GSUM
Cf2py intent(in) n	
Cf2py intent(in) t
Cf2py intent(in) alpha
Cf2py intent(in) beta
Cf2py intent(out) g

      DO S = 1, T
      	GSUM = 0.0d0
      	DO I = 1, N
      		G(S,I) = ALPHA(S,I) * BETA(S,I)
      		GSUM = gsum + G(S,I)
      	ENDDO
      	DO I = 1, N
      		G(S, I) = G(S, I) / GSUM
      	ENDDO
      ENDDO
      END
      
      SUBROUTINE COMPUTEXI(A,B,O,ALPHA,BETA,XI,T,N,M)
      INTEGER T, N, M, I, J, S
      REAL*8 A(N,N), B(N,M)
      INTEGER O(T)
      REAL*8 alpha(T,N), beta(T,N), XI(T-1,N,N)
      REAL*8 SUM
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) t
Cf2py intent(in) o
Cf2py intent(in) alpha
Cf2py intent(in) beta
Cf2py intent(out) xi

      DO S = 1, T-1
      	SUM = 0.0d0
      	DO J = 1, N
      		DO I = 1, N
      			XI(S,I,J) = ALPHA(S,I)*A(I,J)*B(J,O(S+1)+1)*BETA(S+1,J)
      			SUM = SUM + XI(S,I,J)
      		ENDDO
      	ENDDO
      	DO J = 1, N
      		DO I = 1, N
      			XI(S,I,J) = XI(S,I,J) / SUM
      		ENDDO
      	ENDDO
      ENDDO
      END

      SUBROUTINE COMPUTENOMA(A,B,O,ALPHA,BETA,NOMA,T,N,M)
      INTEGER T, N, M, I, J, S
      REAL*8 A(N,N), B(N,M), ALPHA(T,N), BETA(T,N)
      REAL*8 NOMA(N,N), TMP(N,N)
      INTEGER O(T)
      REAL*8 SUM
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) t
Cf2py intent(in) o
Cf2py intent(in) alpha
Cf2py intent(in) beta
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(out) noma
      DO S = 1, T-1
      	SUM = 0.0D0
      	DO J = 1, N
      		DO I = 1, N
      			TMP(I,J) = ALPHA(S,I)*A(I,J)*B(J,O(S+1)+1)*BETA(S+1,J)
      			SUM = SUM + TMP(I,J)
      		ENDDO
      	ENDDO
      	DO J = 1, N
      		DO I = 1, N
      			NOMA(I,J) = NOMA(I,J) + TMP(I,J) / SUM
      		ENDDO
      	ENDDO
      ENDDO
      END

      SUBROUTINE COMPUTEDENOMA(GAMMA,DENOMA,T,N)
      INTEGER T, N, I, S
      REAL*8 GAMMA(T,N), DENOMA(N)
Cf2py intent(in) n
Cf2py intent(in) t
Cf2py intent(in) gamma
Cf2py intent(out) denoma
      DO I = 1, N
      	DENOMA(I) = 0.0D0
      	DO S = 1, T-1
      		DENOMA(I) = DENOMA(I) + GAMMA(S,I)
      	ENDDO
      ENDDO
      END

      SUBROUTINE COMPUTENOMB(O,GAMMA,NOMB,T,N,M)
      INTEGER T, N, M, S, I, K
      REAL*8 O(T), GAMMA(T, N)
      REAL*8 NOMB(N, M)
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) t
Cf2py intent(in) gamma
Cf2py intent(in) o
Cf2py intent(out) nomb
      DO I = 1, N
      	DO K = 1, M
      		NOMB(I,K) = 0.0D0
      		DO S = 1, T
      			IF (O(S) == K-1) THEN
      				NOMB(I,K) = NOMB(I,K) + GAMMA(S,I)
      			ENDIF
      		ENDDO
      	ENDDO
      ENDDO
      END

      subroutine update(A,B,pi,O,gamma,xi,T,N,M)
      integer T, N, M, s, i, j, k
      real*8 A(N,N), B(N, M), pi(N)
      integer O(T)
      real*8 gamma(T,N), xi(T-1,N,N)
      real*8 sum
Cf2py intent(in) t
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(in) o
Cf2py intent(out) a
Cf2py intent(out) b
Cf2py intent(out) pi
Cf2py intent(in) gamma
Cf2py intent(in) xi

      do i = 1, N
      	pi(i) = gamma(1, i)
      enddo

      do i = 1, N
      	sum = 0.0d0
      	do s = 1, T-1
      		sum = sum + gamma(s, i)
      	enddo
      	do j = 1, N
      		A(i,j) = 0.0d0
      		do s = 1, T-1
      			A(i,j) = A(i,j) + xi(s,i,j)
      		enddo
      		A(i,j) = A(i,j) / sum
      	enddo
      	sum = sum + gamma(T,i)
      	do k = 1, M
      		B(i,k) = 0.0d0
      		do s = 1, T
      			if (O(s) == k-1) then
      				B(i,k) = B(i,k) + gamma(s, i)
      			endif
      		enddo
      		B(i,k) = B(i,k) / sum
      	enddo
      enddo
      end
