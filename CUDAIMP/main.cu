
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




const __restrict__ Board BoardIn;



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
            //printf("[%d] ",(*input)[i][j]);
        }
        // printf("\r\n");
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
    for(uint8_t i=0; i < HEIGHT; i++){
        for(uint8_t j=0; j < WIDTH;j++) {
            if ((*output)[i][j] == 0) {
                fail = false;
            }
            if ((*output)[i][j] != (*input)[i][j]) {
                changed = true;
            }
        }
    }

    if(fail){
        return boardFull;
    }
    else if(!changed){
        return boardUnchanged;
    }
    randGen(output);
    return boardUpdated;

}

/**
 * This function adds the random move to the board. This will most likely change later on to fit with the CUDA program so they produce the same results.
 * @param movedBoard A pointer to a Board object to have a random tile added to the board.
 */
__device__ void randGen(Board * movedBoard){
    unsigned long long seed= (*movedBoard)[0][0] + 2 * (*movedBoard)[0][1] + 3 * (*movedBoard)[0][2] + 4 * (*movedBoard)[0][3] + 5 * (*movedBoard)[1][0] + 6 * (*movedBoard)[1][1] + 7 * (*movedBoard)[1][2] + 8 * (*movedBoard)[1][3];
    curandState_t *state;
    curand_init (seed, 0,0, curandState_t *state);

    unsigned int randNum = curand(state);
    unsigned char position = randNum % SIZE;
    while((*movedBoard)[(randNum/WIDTH)][(randNum%HEIGHT)] != 0){
        unsigned int randNum = curand(state);
        unsigned char position = randNum % SIZE;
    }
    unsigned int randomValue = curand(state);
    if(randomValue % 10 == 9){
        (*movedBoard)[(randNum/WIDTH)][(randNum%HEIGHT)] = 4;
    }
    else{
        (*movedBoard)[(randNum/WIDTH)][(randNum%HEIGHT)] = 2;
    }


}


__global__ void kernel(int * scoreList){
    tx = threadIdx.x;
    bx = blockIdx.x;
    bd = blockDim.x;

    uint32_t threadNum = bx * bd + tx;

    Board board;
    int i,j;
    for(i = 0; i < HEIGHT; i++){
        for(j = 0; j < WIDTH; j++){
            board[i][j] = BoardIn[i][j];
        }
    }

    status stat;
    Move mList[NUMMOVES];

    mList[0] = (threadNum & m0Mask);
    mList[1] = (threadNum & m1Mask);
    mList[2] = (threadNum & m2Mask);
    mList[3] = (threadNum & m3Mask);
    mList[4] = (threadNum & m4Mask);
    mList[5] = (threadNum & m5Mask);
    mList[6] = (threadNum & m6Mask);
    mList[7] = (threadNum & m7Mask);

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


    }
    __syncthreads();
    return;
}




int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
