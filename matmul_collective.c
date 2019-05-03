/* Parallel matrix multiplication using block partition method
 and MPI Collective

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
	int i, j;
	double *A, *B, *C, *recvA, *recvB, *sendC, *bufA, *bufB;

	/*
	each process holding a pair matrices bufA, bufB
	recvA, recvB is to store the bufA, bufB received from others
	the mini-matmul operation will be execute b times and sum the result to the sendC buffer
	*/
	recvA = (double *)malloc(blockDim*blockDim*sizeof(double));
	recvB = (double *)malloc(blockDim*blockDim*sizeof(double));
	sendC = (double *)malloc(blockDim*blockDim*sizeof(double));
	bufA = (double *)malloc(blockDim*blockDim*sizeof(double));
	bufB = (double *)malloc(blockDim*blockDim*sizeof(double));


	/* initialization */
	if (rank == 0) {
		A = (double *)malloc(n*n*sizeof(double));
		B = (double *)malloc(n*n*sizeof(double));
		C = (double *)malloc(n*n*sizeof(double));

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				A[n*i + j] = j;
				B[n*i + j] = j;
				C[n*i + j] = 0;
			}
		}
	}
	for (i = 0; i < blockDim*blockDim; i++) {
		sendC[i] = 0;
	}


	/*
	divide matrices A, B into b*b blocks,
	each will be sent to the respective processors and stored in bufA, bufB
	*/
	MPI_Datatype BlockType, ColType;
	MPI_Type_vector(blockDim, blockDim, n, MPI_DOUBLE, &ColType);
	MPI_Type_create_resized(ColType, 0, sizeof(double), &BlockType);
	MPI_Type_commit(&BlockType);

	int *displacement = (int *)malloc(numProc*sizeof(int));
	int *counts = (int *)malloc(numProc*sizeof(int));
	for (i = 0; i < b; i++) {
		for (j = 0; j < b; j++) {
			displacement[b*i + j] = n*blockDim*i + blockDim*j;
			counts[b*i + j] = 1;
		}
	}

	MPI_Scatterv(A, counts, displacement, BlockType, bufA, blockDim*blockDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(B, counts, displacement, BlockType, bufB, blockDim*blockDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);


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


	// /* join the results */
	MPI_Gatherv(sendC, blockDim*blockDim, MPI_DOUBLE, C, counts, displacement, BlockType, 0, MPI_COMM_WORLD);


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
