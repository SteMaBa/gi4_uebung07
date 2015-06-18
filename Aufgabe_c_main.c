#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "init_matrix.h"
#include <pthread.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#define MATRIX_SIZE (1024)
#define NUM_THREADS 4

double **A;
double *b;
double *X;
double *X_old;
double *temp;


void * doJacobi (void * arg)
{
	int i_thread = *((int *) arg);	//Anzahl Threads wurde übergeben
	int begin = i_thread*MATRIX_SIZE/NUM_THREADS;  	//Untere Grenze i, berechnet aus Threadnummer in arg
	int end = (i_thread+1)*MATRIX_SIZE/NUM_THREADS;  //Obere Grenze i
	int j = 0;				//Zähler j=0
	int i = 0;				//Zähler i=0
	double xi = 0.0;		//Temporäres xi=0
    __m128 m1, m2, m3, m4;

	//Jacobi-Algorithmus
	for (i = begin; i < end; i++) {

		for (j = 0, xi = 0.0; j < i; j++)
				xi += A[i][j] * X_old[j];	//Variablen global definiert
			for (j = i + 1; j < MATRIX_SIZE; j++)
				xi += A[i][j] * X_old[j];



		//X[i] = (b[i] - xi) / A[i][i];

	    //Speichere b[i] in m1, xi in m2, A[i][i] in m3

		m1[i] = b[i];
		m2[i] = xi;
		m3[i] = A[i][i];

	}

    	//Rechne m1 - m2 und anschließend / m3 in m4
		m4 = _mm_sub_ps(m2,m1);
		m4 = _mm_div(m4, m3);

		//Kopiere aus m4 in X[i]

		for (i = begin; i < end; i++)
		{
			X[i] = m4[i];
		}


	return NULL;
}

int main(int argc, char **argv)
{
	unsigned int i, j;
	unsigned int iterations = 0;
	double error, xi, norm, max = 0.0;
	struct timeval start, end;

	pthread_t threads[NUM_THREADS];	//Threads bereitstellen
	__m128d *av, *bv, *cv;
	double vsum = 0.0;



	printf("\nInitialize system of linear equations...\n");
	/* allocate memory for the system of linear equations */
	init_matrix(&A, &b, MATRIX_SIZE);
	X = (double *)malloc(sizeof(double) * MATRIX_SIZE);
	X_old = (double *)malloc(sizeof(double) * MATRIX_SIZE);

	/* a "random" solution vector */
	for (i = 0; i < MATRIX_SIZE; i++) {
		X[i] = ((double)rand()) / ((double)RAND_MAX) * 10.0;
		X_old[i] = 0.0;
	}

	printf("Start Jacobi method...\n");

	gettimeofday(&start, NULL);




	/* Jacobi iterations */
		while (1) {
			iterations++;

			temp = X_old;
			X_old = X;
			X = temp;



			//Jacobi starten
			for (i = 0; i < NUM_THREADS; i++)
			{

				//Threads erstellen, obere und untere Grenze für einen Thread werden in doJacobi berechnet
				pthread_create (&threads[i], NULL, doJacobi, (void*) &i);
			}

			//Auf alle Threads warten, um deterministische Werte bei Zugriff auf die globalen Variablen sicherzustellen
			for (i = 0; i < NUM_THREADS; i++)
			{
				pthread_join(threads[i], NULL);
			}


			//Mit SSE-Instruktionen
			if (iterations % 500 == 0) {	/* calculate the Euclidean norm between X_old and X */
				norm = 0.0;

				//Bewege X_old - X in av und bv
				av = (__m128d*)(X_old-X);
				bv = (__m128d*)(X_old-X);

				//Multipliziere av und bv in cv
				cv = _mm_mul_ps (av, bv);

				//Summiere alle Einträge aus cv in norm
				for (j = 0; j < 128; j++)
				{
					norm += cv[j];
				}

				//Abbruchbedingung prüfen
				if (norm < 0.0000001)
				{
					break;
				}

			}
		}

		//Jacobi iterations end


	gettimeofday(&end, NULL);

	if (MATRIX_SIZE < 16) {
		printf("Print the solution...\n");
		/* print solution */
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++)
				printf("%8.2f\t", A[i][j]);
			printf("*\t%8.2f\t=\t%8.2f\n", X[i], b[i]);
		}
	}

	printf("Check the result...\n");
	/* 
	 * check the result 
	 * X[i] have to be 1
	 */
	for (i = 0; i < MATRIX_SIZE; i++) {
		error = fabs(X[i] - 1.0f);

		if (max < error)
			max = error;
		if (error > 0.01f)
			printf("Result is on position %d wrong (%f != 1.0)\n",
			       i, X[i]);
	}
	printf("maximal error is %f\n", max);

	printf("\nmatrix size: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
	printf("number of iterations: %d\n", iterations);
	printf("Time : %lf sec\n",
	       (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec -
							      start.tv_usec) /
	       1000000.0);

	/* frees the allocated memory */
	free(X_old);
	free(X);
	clean_matrix(&A);
	clean_vector(&b);

	return 0;
}
