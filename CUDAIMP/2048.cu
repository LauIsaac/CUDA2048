
#include <wb.h>
#include "types.h"
#include <stdio.h>
#include <stdbool.h>
#include <curand_kernel.h>


#define m0Mask 0xC000
#define m1Mask 0x3000
#define m2Mask 0x0C00
#define m3Mask 0x0300
#define m4Mask 0x00C0
#define m5Mask 0x0030
#define m6Mask 0x000C
#define m7Mask 0x0003



#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)







/**
 * This function returns a score for the Board. Right now just returns the game score, may be worked to include neighboring tiles combined scores if we need more accuracy.
 * @param input Pointer to Board to be scored.
 * @return The score
 */
__device__ uint32_t score(Board * input){
    uint32_t scoreVal;
    scoreVal = 0;
    for(uint8_t i=0; i < HEIGHT; i++){
        for(uint8_t j=0; j < WIDTH; j++){
            if((*input)[i][j] == 2){
                scoreVal += 2;
            }
            else if((*input)[i][j] != 0){
                scoreVal += (*input)[i][j]+(((*input)[i][j])/2);
            }
            printf("[%d] ",(*input)[i][j]);
        }
        printf("\r\n");
    }
    return scoreVal;
}

__device__ void leftSolver(Board * output){
    int8_t i, j, moveCounter, mergeCounter;

    //This section moves all items through the 0's.
    moveCounter = 0;
    for(i=0; i < HEIGHT; i++){
        for(j=0; j < WIDTH; j++){
            int test = (*output)[i][j];
            if(test == 0){
                moveCounter++;
            }
            else if(moveCounter != 0){
                ((*output))[i][(j-moveCounter)] = ((*output))[i][j];
                ((*output))[i][j] = 0;
            }
        }
        moveCounter = 0;
    }

    //This section merges any nearby values
    mergeCounter = 0;
    for(i=0; i < HEIGHT; i++){
        for(j=0; j < WIDTH-1; j++){
            if(((*output))[i][j] == ((*output))[i][j+1]){
                ((*output))[i][j] = 2*(((*output))[i][j]);
                mergeCounter++;
                ((*output))[i][j+1] = 0;
            }
            if(mergeCounter != 0 && ((*output))[i][j+1] != 0){
                (*output)[i][j] = (*output)[i][j+1];
                (*output)[i][j+1] = 0;
            }
        }
        mergeCounter = 0;
    }



}


__device__ void rightSolver(Board * output){
    int8_t i, j, moveCounter, mergeCounter;

    //This section moves all items through the 0's.
    moveCounter = 0;
    for(i=0; i < HEIGHT; i++){
        for(j=WIDTH-1; j >= 0; j--){
            int test = (*output)[i][j];
            if(test == 0){
                moveCounter++;
            }
            else if(moveCounter != 0){
                ((*output))[i][(j+moveCounter)] = ((*output))[i][j];
                ((*output))[i][j] = 0;
            }
        }
        moveCounter = 0;
    }

    //This section merges any nearby values
    mergeCounter = 0;
    for(i=0; i < HEIGHT; i++){
        for(j=WIDTH-1; j > 0; j--){
            if(((*output))[i][j] == ((*output))[i][j-1]){
                ((*output))[i][j] = 2*(((*output))[i][j]);
                mergeCounter++;
                ((*output))[i][j-1] = 0;
            }
            if(mergeCounter != 0 && (*output)[i][j-1] != 0){
                (*output)[i][j] = (*output)[i][j-1];
                (*output)[i][j-1] = 0;
            }
        }
        mergeCounter = 0;
    }
}


__device__ void upSolver(Board * output){
    int8_t i, j, moveCounter, mergeCounter;

    //This section moves all items through the 0's.
    moveCounter = 0;
    for(j=0; j < WIDTH; j++){
        for(i=0; i < HEIGHT; i++){
            if((*output)[i][j] == 0){
                moveCounter++;
            }
            else if(moveCounter != 0){
                (*output)[(i-moveCounter)][j] = (*output)[i][j];
                (*output)[i][j] = 0;
            }
        }
        moveCounter = 0;
    }

    mergeCounter = 0;
    for(j=0; j < WIDTH; j++){
        for(i=0; i < HEIGHT-1; i++){
            if((*output)[i][j] == (*output)[i+1][j]){
                (*output)[i][j] = 2*((*output)[i][j]);
                mergeCounter++;
                (*output)[i+1][j] = 0;
            }
            if(mergeCounter != 0 && (*output)[i+1][j] != 0){
                (*output)[i][j] = (*output)[i+1][j];
                (*output)[i+1][j] = 0;
            }
        }
        mergeCounter = 0;
    }

}



__device__ void downSolver(Board * output){
    int8_t i, j, moveCounter, mergeCounter;

    //This section moves all items through the 0's.
    //Might not need to dereference board pointers
    moveCounter = 0;
    for(j=0; j < WIDTH; j++){
        for(i=HEIGHT-1; i >= 0; i--){
            if((*output)[i][j] == 0){
                moveCounter++;
            }
            else if(moveCounter != 0){
                (*output)[(i+moveCounter)][j] = (*output)[i][j];
                (*output)[i][j] = 0;
            }
        }
        moveCounter = 0;
    }
    mergeCounter = 0;
    for(j=0; j < WIDTH; j++){
        for(i=HEIGHT-1; i > 0; i--){
            if((*output)[i][j] == (*output)[i-1][j]){
                (*output)[i][j] = 2*((*output)[i][j]);
                mergeCounter++;
                (*output)[i-1][j] = 0;
            }
            if(mergeCounter != 0 && (*output)[i-1][j] != 0){
                (*output)[i][j] = (*output)[i-1][j];
                (*output)[i-1][j] = 0;
            }
        }
        mergeCounter = 0;
    }

}

/**
 * This function adds the random move to the board. This will most likely change later on to fit with the CUDA program so they produce the same results.
 * @param movedBoard A pointer to a Board object to have a random tile added to the board.
 */
 __device__ void randGen(Board * movedBoard){
    unsigned long long seed= (*movedBoard)[0][0] + 2 * (*movedBoard)[0][1] + 3 * (*movedBoard)[0][2] + 4 * (*movedBoard)[0][3] + 5 * (*movedBoard)[1][0] + 6 * (*movedBoard)[1][1] + 7 * (*movedBoard)[1][2] + 8 * (*movedBoard)[1][3];
    curandState_t *state;
    curand_init(seed, 0,0, state);

    unsigned int randNum = curand(state);
    unsigned char position = randNum % SIZE;
    while((*movedBoard)[(position/WIDTH)][(position%HEIGHT)] != 0){
        randNum = curand(state);
        position = randNum % SIZE;
    }
    unsigned int randomValue = curand(state);
    if(randomValue % 10 == 9){
        (*movedBoard)[(position/WIDTH)][(position%HEIGHT)] = 4;
    }
    else{
        (*movedBoard)[(position/WIDTH)][(position%HEIGHT)] = 2;
    }


}


/**
 *This takes the predetermined move and returns a Board that has had that move applied. This should be the link between the recursive section of the code and the solver
 * @param input The board that is requested to be solved
 * @param currMove The move to apply to the board
 * @param output A pointer for the board after the move has occurred to be stored in.
 * @return Returns the status of the move. Whether or not the board was updated.
 */
__device__ status moveHandler(Board *input, Move currMove){

    

    switch(currMove){
        case(up):
            // printf("Moving up \r\n");
            upSolver(input);
            break;
        case down:
            // printf("Moving down \r\n");
            downSolver(input);
            break;
        case left:
            // printf("Moving left \r\n");
            leftSolver(input);
            break;
        case right:
            // printf("Moving right \r\n");
            rightSolver(input);
            break;

    }

    bool changed = false;
    bool fail = true;

    for(uint8_t i=0; i < HEIGHT; i++){
        for(uint8_t j=0; j < WIDTH;j++) {
            if ((*input)[i][j] == 0) {
                fail = false;
            }
            //if ((*checkBoard)[i][j] != (*input)[i][j]) {
            //    changed = true;
            //}
        }
    }

    if(fail){
        return boardFull;
    }
    else if(!changed){
        return boardUnchanged;
    }
    randGen(input);
    return boardUpdated;

}




__global__ void kernel(Board *BoardIn, int * scoreList){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bd = blockDim.x;

    uint32_t threadNum = bx * bd + tx;

    Board board;
    int i,j;
    for(i = 0; i < HEIGHT; i++){
        for(j = 0; j < WIDTH; j++){
            board[i][j] = (BoardIn)[i][j];
        }
    }

    status stat;
    Move mList[NUMMOVES];

    //Bitwise and with mask ends up creating many invalid moves (see: 0xC000 & 0xC000), need to rightshift
    mList[0] = (Move) (threadNum & m0Mask);
    mList[1] = (Move) (threadNum & m1Mask);
    mList[2] = (Move) (threadNum & m2Mask);
    mList[3] = (Move) (threadNum & m3Mask);
    mList[4] = (Move) (threadNum & m4Mask);
    mList[5] = (Move) (threadNum & m5Mask);
    mList[6] = (Move) (threadNum & m6Mask);
    mList[7] = (Move) (threadNum & m7Mask);

    scoreList[threadNum] = 0;
    for(i = 0; i < NUMMOVES; i++){
        stat = moveHandler(&board,mList[i]);
        if(stat != boardUpdated){
            break;
        }
        if(i != 7){
            randGen(&board);
        }
        else{
            scoreList[threadNum] = score(&board);
        }
        if(threadNum == 0){
            printf("LIST %d: ",i);
            for(int q = 0; i < HEIGHT; i++){
                for(int r = 0; j < WIDTH; j++){
                    printf("%d ",board[q][r]);
                }
            }
            printf("\r\n");
        }


    }
    if(scoreList[threadNum] != 0){
        printf("DEBUG SCORE:%d\r\n",scoreList[threadNum]);
    }
    __syncthreads();
    return;
}




int main(int argc, char **argv) {
    wbArg_t arg;
    Board hostInputBoard;
    Board * deviceInputBoard;
    char *inputBoardFile;
    int *hostScoreList;
    int *deviceScoreList;
    int Score;
    int inputLength;
        
    arg = wbArg_read(argc, argv);
    
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");


    int numScores = (int)pow(4, 8);
    int scoreListSize = numScores * sizeof(int);
    int boardSize = SIZE * sizeof(int);
    
    //inputBoardFile = (char *)wbImport(wbArg_getInputFile(arg, 0), &inputLength);
    hostScoreList = (int *)malloc(scoreListSize);
    /*
    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            (*hostInputBoard)[i][j] = (*inputBoardFile)[i * WIDTH + j];
        }
    }
    */

    
    hostInputBoard[0][0] = 0;
    hostInputBoard[0][1] = 0;
    hostInputBoard[0][2] = 0;
    hostInputBoard[0][3] = 0;
    hostInputBoard[1][0] = 2;
    hostInputBoard[1][1] = 0;
    hostInputBoard[1][2] = 0;
    hostInputBoard[1][3] = 0;
    hostInputBoard[2][0] = 0;
    hostInputBoard[2][1] = 2;
    hostInputBoard[2][2] = 0;
    hostInputBoard[2][3] = 0;
    hostInputBoard[3][0] = 0;
    hostInputBoard[3][1] = 0;
    hostInputBoard[3][2] = 0;
    hostInputBoard[3][3] = 0;
    

    wbCheck(cudaMalloc((void**)&deviceScoreList, scoreListSize)); 
    wbCheck(cudaMalloc((void**)&deviceInputBoard, boardSize)); 

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    
    wbCheck(cudaMemcpy(deviceInputBoard, &hostInputBoard, boardSize, cudaMemcpyHostToDevice));


    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");

    dim3 DimGrid(256, 1, 1);
    dim3 DimBlock(256, 1, 1);
    kernel<<<DimGrid, DimBlock>>>(deviceInputBoard,deviceScoreList);

    wbTime_stop(Compute, "Doing the computation on the GPU");
    
    cudaDeviceSynchronize();
    wbCheck(cudaPeekAtLastError());
    ////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostScoreList, deviceScoreList, scoreListSize, cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    for(uint32_t num = 0; num < numScores; num++){\
        if(hostScoreList[num] != 0){
            printf("Score %d is %d \r\n", num, hostScoreList[num]);
        }
    }

    wbSolution(arg, hostScoreList, scoreListSize);

    wbCheck(cudaFree(deviceScoreList));

    return 0;
}
