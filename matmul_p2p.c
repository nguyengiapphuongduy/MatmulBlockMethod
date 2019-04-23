/* Parallel matrix multiplication using block partition method
 and MPI point-to-point communication

@author: 1610473 Nguyen Giap Phuong Duy

compile:
mpicc -o mm_run matmul_p2p.c -lm

execute:
mpiexec -n 16 ./mm_run 100

INPUT:
1 2 3 ... 99 100
1 2 3 ... 99 100
. . . ... .. ...
. . . ... .. ...
. . . ... .. ...
1 2 3 ... 99 100

assume that all matrices in this program are square
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TAG_IGNORE 0

void matmul(double *ina, double *inb, double *outc, int dim) {
	int i, j, k;
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			for (k = 0; k < dim; k++) {
				outc[dim*i + j] += ina[dim*i + k] * inb[dim*k + j];
			}
		}
	}
}

void partition(double *src, double *dest, int r, int c, int ndim, int bdim) {
	int i, j, offsetRow = r*bdim, offsetCol = c*bdim;
	for (i = 0; i < bdim; i++) {
		for (j = 0; j < bdim; j++) {
			dest[bdim*i + j] = src[ndim*(i + offsetRow) + (j + offsetCol)];
		}
	}
}

void join(double *src, double *part, int r, int c, int ndim, int bdim) {
	int i, j, offsetRow = r*bdim, offsetCol = c*bdim;
	for (i = 0; i < bdim; i++) {
		for (j = 0; j < bdim; j++) {
			src[ndim*(i + offsetRow) + (j + offsetCol)] = part[bdim*i + j];
		}
	}
}


int main(int argc, char *argv[]) {

	int n = 100;
	if (argc > 1) {
		n = atoi(argv[1]);
	}

	int rank, numProc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);

	int b = ceil(sqrt(numProc));
	if (b*b != numProc) {
		if (rank == 0) {
			printf("Number of process must be square number\n");
		}
		MPI_Finalize();
		return 1;
	}
	if (n % b != 0) {
		if (rank == 0) {
			printf("The block dimension must be a divisor of the matrix dimension. Check the number of processors\n");
		}
		MPI_Finalize();
		return 1;
	}

	int blockDim = n/b;
	int i, j, offsetRow, offsetCol;
	double *A, *B, *C, *recvA, *recvB, *sendC, *bufA, *bufB, *bufC;

	/*
	each process holding a pair matrices bufA, bufB
	recvA, recvB is to store the bufA, bufB received from others
	the mini-matmul operation will be execute b times and sum the result to the sendC buffer
	bufC is to concatenate the final result
	*/
	recvA = (double *)malloc(blockDim*blockDim*sizeof(double));
	recvB = (double *)malloc(blockDim*blockDim*sizeof(double));
	sendC = (double *)malloc(blockDim*blockDim*sizeof(double));
	bufA = (double *)malloc(blockDim*blockDim*sizeof(double));
	bufB = (double *)malloc(blockDim*blockDim*sizeof(double));
	bufC = (double *)malloc(blockDim*blockDim*sizeof(double));

	if (rank == 0) {
		/* initialization */
		A = (double *)malloc(n*n*sizeof(double));
		B = (double *)malloc(n*n*sizeof(double));
		C = (double *)malloc(n*n*sizeof(double));

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				A[n*i + j] = j + 1;
				B[n*i + j] = j + 1;
				C[n*i + j] = 0;
			}
		}

		/*
		divide matrices A, B into b*b blocks,
		each will be sent to the respective processors and stored in bufA, bufB
		*/
		partition(A, bufA, 0, 0, n, blockDim);
		partition(B, bufB, 0, 0, n, blockDim);
		double *sendbuf = (double *)malloc(blockDim*blockDim*sizeof(double));
		for (i = 0; i < b; i++) {
			for (j = 0; j < b; j++) {
				if (i + j == 0) continue;
				partition(A, sendbuf, i, j, n, blockDim);
				MPI_Send(sendbuf, blockDim*blockDim, MPI_DOUBLE, b*i + j, TAG_IGNORE, MPI_COMM_WORLD);
				partition(B, sendbuf, i, j, n, blockDim);
				MPI_Send(sendbuf, blockDim*blockDim, MPI_DOUBLE, b*i + j, TAG_IGNORE, MPI_COMM_WORLD);
			}
		}
	} else {
		/* respectively receive blocks from root */
		MPI_Recv(bufA, blockDim*blockDim, MPI_DOUBLE, 0, TAG_IGNORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(bufB, blockDim*blockDim, MPI_DOUBLE, 0, TAG_IGNORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	for (i = 0; i < blockDim; i++) {
		for (j = 0; j < blockDim; j++) {
			sendC[blockDim*i + j] = 0;
		}
	}

	/* start calculation */
	double t = MPI_Wtime();

	/* send bufA to the others in the same row, and bufB to the others in the same column */
	for (i = 0; i < b; i++) {
		int row = rank/b, col = rank%b;
		if (i != col) { // send to the others in the same row but not itself
			MPI_Send(bufA, blockDim*blockDim, MPI_DOUBLE, b*row + i, TAG_IGNORE, MPI_COMM_WORLD);
		}
		if (i != row) { // send to the others in the same column but not itself
			MPI_Send(bufB, blockDim*blockDim, MPI_DOUBLE, b*i + col, TAG_IGNORE, MPI_COMM_WORLD);
		}
	}

	/* respectively receive bufA, bufB */
	for (i = 0; i < b; i++) {
		int row = rank/b, col = rank%b;
		if (i != col) { // receive from the others in the same row but not itself
			MPI_Recv(recvA, blockDim*blockDim, MPI_DOUBLE, b*row + i, TAG_IGNORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			memcpy(recvA, bufA, blockDim*blockDim*sizeof(double));
		}
		if (i != row) { // receive from the others in the same column but not itself
			MPI_Recv(recvB, blockDim*blockDim, MPI_DOUBLE, b*i + col, TAG_IGNORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			memcpy(recvB, bufB, blockDim*blockDim*sizeof(double));
		}
		matmul(recvA, recvB, sendC, blockDim);
	}

	/* join the results */
	if (rank != 0) {
		MPI_Send(sendC, blockDim*blockDim, MPI_DOUBLE, 0, TAG_IGNORE, MPI_COMM_WORLD);
	} else {
		for (i = 0; i < b; i++) {
			for (j = 0; j < b; j++) {
				if (i + j == 0) {
					join(C, sendC, i, j, n, blockDim);
				} else {
					MPI_Recv(bufC, blockDim*blockDim, MPI_DOUBLE, b*i + j, TAG_IGNORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					join(C, bufC, i, j, n, blockDim);
				}
			}
		}
	}

	/* print result */
	if (rank == 0) {
		printf("Matrix multiplication result for square matrix with [%dx%d] dimension:\n", n, n);
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				printf("%.0f\t", C[n*i + j]);
			}
			printf("\n");
		}
		printf("Finish calculation in %f seconds.\n", MPI_Wtime() - t);
	}

	MPI_Finalize();
	return 0;
}