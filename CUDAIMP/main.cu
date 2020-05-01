
#include <wb.h>
#include "types.h"
#include <stdio.h>
#include <stdbool.h>


void upSolver(Board * input, Board * output);
void downSolver(Board * input, Board * output);
void leftSolver(Board * input, Board * output);
void rightSolver(Board * input, Board * output);
void randGen(Board * movedBoard);
status moveHandler(Board *input, Move currMove, Board * output);
uint32_t score(Board * input);

void makeMoveList(Board * bIn, MoveList predefinedMoves, MoveList moves, uint8_t end, uint8_t index, uint8_t r);

/**
 * This will be removed and turned into the kernel
 */
void makeMoveList(Board * bIn, MoveList predefinedMoves, MoveList moves, uint8_t end, uint8_t index, uint8_t r){
    if(index == r){
        Board * moveBoard;
        moveBoard = malloc(sizeof(Board));
        //TODO: This is dirty and should be changed 100% so as not to have 2 memcpys as its copying it here and later on, but that can be gotten to later.
        memcpy(moveBoard,bIn,sizeof(Board));
        status sOut;

        uint8_t ref = moves[0];

        uint8_t j;
        for(j = 0; j < r; j++) {
            sOut = moveHandler(moveBoard, moves[j], moveBoard);
            if (sOut == boardFull) {
                MoveScores[ref][MoveTracker[ref]] = 0;
                //printf("Board with move [%d,%d,%d,%d,%d] is full af \r\n",moves[0],moves[1],moves[2],moves[3],moves[4]);
                MoveTracker[ref]++;
                free(moveBoard);
                return;
            }
        }
        uint32_t boardScore = score(moveBoard);
        //printf("Board with move [%d,%d,%d,%d,%d] had a score of %d \r\n",moves[0],moves[1],moves[2],moves[3],moves[4],boardScore);
        MoveScores[ref][MoveTracker[ref]] = boardScore;
        MoveTracker[ref]++;
        free(moveBoard);
        return;
    }

    for(int i = 0; i <=end; i++){
        moves[index] = predefinedMoves[i];
        makeMoveList(bIn, predefinedMoves, moves, end, index+1, r);
    }
}




/**
 *This takes the predetermined move and returns a Board that has had that move applied. This should be the link between the recursive section of the code and the solver
 * @param input The board that is requested to be solved
 * @param currMove The move to apply to the board
 * @param output A pointer for the board after the move has occurred to be stored in.
 * @return Returns the status of the move. Whether or not the board was updated.
 */
__device__ status moveHandler(Board *input, Move currMove, Board * output){
    switch(currMove){
        case(up):
            // printf("Moving up \r\n");
            upSolver(input,output);
            break;
        case down:
            // printf("Moving down \r\n");
            downSolver(input,output);
            break;
        case left:
            // printf("Moving left \r\n");
            leftSolver(input,output);
            break;
        case right:
            // printf("Moving right \r\n");
            rightSolver(input,output);
            break;
        default:
            exit(EXIT_FAILURE);

    }

    bool fail = true;
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

__device__ void leftSolver(Board *input, Board * output){
    //Boilerplate code to transfer the input Board into the output.
    memcpy(output,input,sizeof(Board));
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


__device__ void rightSolver(Board * input, Board * output){
    //Boilerplate code to transfer the input Board into the output.
    memcpy(output,input,sizeof(Board));
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


__device__ void upSolver(Board * input, Board * output){
    //Boilerplate code to transfer the input Board into the output.
    memcpy(output,input,sizeof(Board));
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



__device__ void downSolver(Board * input, Board * output){
    //Boilerplate code to transfer the input Board into the output.
    memcpy(output,input,sizeof(Board));
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
 * This function adds the random move to the board. This will most likely change later on to fit with the CUDA program so they produce the same results.
 * @param movedBoard A pointer to a Board object to have a random tile added to the board.
 */
void randGen(Board * movedBoard){
    //TODO Add in generating new moves. Need to figure out how to do random on CUDA
}




int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
