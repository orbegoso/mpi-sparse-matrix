/*  
 *  Brian Orbegoso
 *  CPSC 474
 *  December 18, 2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"

#define MAX_ELEMENTS    50000
#define ROOT            0

struct Element {
    int x;
    int y;
    int value;
};

int main(int argc, char* argv[]) {
    int rank;
    int size;

    int numRows = 0;
    int numCols = 0;

    int sum = 0;
    int rem;

    int numElems = 0;

    int *sendCounts;
    int *displacements;

    struct Element rbufElems[MAX_ELEMENTS];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sendCounts = malloc(size * sizeof *sendCounts);
    displacements = malloc(size * sizeof *displacements);

    /* Create an MPI_Datatype for struct Element */
    MPI_Datatype mpi_element_type;

    int bl[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(struct Element, x);
    offsets[1] = offsetof(struct Element, y);
    offsets[2] = offsetof(struct Element, value);

    MPI_Type_create_struct(3, bl, offsets, types, &mpi_element_type);
    MPI_Type_commit(&mpi_element_type);

    /* Read from file to get numRows and numCols*/
    if (rank == ROOT) {
        FILE* fileArr;
        size_t vecLen;
        char* vec = NULL;

        fileArr = fopen("matrix", "r");

        while (!feof(fileArr)) {
            getline(&vec, &vecLen, fileArr);
            numRows++;
        }

        while (*vec) {
            if (*vec == ' ') {
                numCols++;
            }

            vec++;
        }

        numCols++;

        fclose(fileArr);
    }

    /* Broadcast numRows and numCols to all the processes */
    MPI_Bcast(&numRows, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&numCols, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    struct Element arrElems[numRows][numCols];

    /* Read from the file and fill arrElems with values */
    if (rank == ROOT) {
        FILE* fileArr;
        fileArr = fopen("matrix", "r");

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                fscanf(fileArr, "%d", &arrElems[i][j].value);
                arrElems[i][j].x = i;
                arrElems[i][j].y = j;
            }
        }

        fclose(fileArr);
    }

    /* Caluclate displacements and sendCounts */
    rem = (numRows * numCols) % size;

    for (int i = 0; i < size; i++) {
        sendCounts[i] = (numRows * numCols) / size;
        if (rem > 0) {
            sendCounts[i]++;
            rem--;
        }

        displacements[i] = sum;
        sum += sendCounts[i];
    }

    /* Send chunks of the matrix arrElems to different processes */
    MPI_Scatterv(&arrElems, sendCounts, displacements, mpi_element_type, &rbufElems, MAX_ELEMENTS, mpi_element_type, ROOT, MPI_COMM_WORLD);

    if (rank != ROOT) {
        for (int i = 0; i < sendCounts[rank]; i++) {
            if (rbufElems[i].value != 0) {
                printf("process %d sends value %d from location (%d, %d) to ROOT\n", rank, rbufElems[i].value, rbufElems[i].x, rbufElems[i].y);
                MPI_Send(&rbufElems[i], 1, mpi_element_type, ROOT, 0, MPI_COMM_WORLD);
                numElems++;
            }
        }

        MPI_Send(&numElems, 1, MPI_INT, ROOT, 1, MPI_COMM_WORLD);

    } else {
        for (int i = 0; i < size - 1; i++) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            numElems += temp;
        }
        
        /* Receive and output the nonzero elements sent from other processes */
        struct Element elem;
        for (int i = 0; i < numElems; i++) {
            MPI_Recv(&elem, 1, mpi_element_type, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("ROOT has received %d from location (%d, %d)\n", elem.value, elem.x, elem.y);
        }

        /* Output the nonzero elements the ROOT was given when arrElems was scattered */
        for (int i = 0; i < sendCounts[rank]; i++) {
            if (rbufElems[i].value != 0) {
                printf("ROOT has %d from location (%d, %d)\n", rbufElems[i].value, rbufElems[i].x, rbufElems[i].y);
            }
        }
    }

    MPI_Type_free(&mpi_element_type);
    MPI_Finalize();

    free(sendCounts);
    free(displacements);

    return 0;
}
