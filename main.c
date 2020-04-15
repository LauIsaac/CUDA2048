#include <stdio.h>
#include "types.h"
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

void upSolver(Board * input, Board * output);
void downSolver(Board * input, Board * output);
void leftSolver(Board * input, Board * output);
void rightSolver(Board * input, Board * output);
void randGen(Board * movedBoard);
status moveHandler(Board *input, Move currMove, Board * output);
uint32_t score(Board * input);

void makeMoveList(Board * bIn, MoveList predefinedMoves, MoveList moves, uint8_t end, uint8_t index, uint8_t r);


//MAKE This number 4^(#Iterations -1)
#define NUMLATEMOVES 4

//Array to hold all moves, seperated by the move which are the first dim of the array
uint32_t MoveScores[NUMMOVES][NUMLATEMOVES];

//Array to keep track of how many of a move have been added to the array of scores.
uint16_t MoveTracker[NUMMOVES];




int main() {
    printf("Hello, World!\n");
    Board Ayye;
    Ayye[0][0] = 0;
    Ayye[0][1] = 0;
    Ayye[0][2] = 0;
    Ayye[0][3] = 0;
    Ayye[1][0] = 2;
    Ayye[1][1] = 0;
    Ayye[1][2] = 0;
    Ayye[1][3] = 0;
    Ayye[2][0] = 256;
    Ayye[2][1] = 2;
    Ayye[2][2] = 0;
    Ayye[2][3] = 2;
    Ayye[3][0] = 256;
    Ayye[3][1] = 16;
    Ayye[3][2] = 2;
    Ayye[3][3] = 2;

    MoveList predefinedList = {up,down,left,right};
    MoveList tempList;
    makeMoveList(&Ayye,predefinedList,tempList,NUMMOVES,0,NUMITERATIONS);

    printf("Completed\r\n");
    while(1){

    }

    return 0;
}


/**
 This function generates all moves with a set recursive depth. It stores all scores for all boards generated.
 * @param bIn The initial board that all moves will start with
 * @param predefinedMoves A list of predefined moves. The algorithm is easier if this is predefined
 * @param moves An empty base array, this is here to hold the moves that are being performed
 * @param end The end of the predefinedMove array
 * @param index Current index of the moves array
 * @param r The max number of iterations to perform
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
                printf("Board with move [%d,%d,%d,%d,%d] is full af \r\n",moves[0],moves[1],moves[2],moves[3],moves[4]);
                MoveTracker[ref]++;
                free(moveBoard);
                return;
            }
        }
        uint32_t boardScore = score(moveBoard);
        printf("Board with move [%d,%d,%d,%d,%d] had a score of %d \r\n",moves[0],moves[1],moves[2],moves[3],moves[4],boardScore);
        MoveScores[ref][MoveTracker[ref]] = 0;
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
status moveHandler(Board *input, Move currMove, Board * output){
    switch(currMove){
        case(up):
            printf("Moving up \r\n");
            upSolver(input,output);
            break;
        case down:
            printf("Moving down \r\n");
            downSolver(input,output);
            break;
        case left:
            printf("Moving left \r\n");
            leftSolver(input,output);
            break;
        case right:
            printf("Moving right \r\n");
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
uint32_t score(Board * input){
    uint32_t scoreVal;
    scoreVal = 0;
    for(uint8_t i=0; i < HEIGHT; i++){
        for(uint8_t j=0; j < WIDTH; j++){
            scoreVal += (*input)[i][j];
            printf("[%d] ",(*input)[i][j]);
        }
        printf("\r\n");
    }
    return scoreVal;
}

void leftSolver(Board *input, Board * output){
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


void rightSolver(Board * input, Board * output){
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


void upSolver(Board * input, Board * output){
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



void downSolver(Board * input, Board * output){
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
    //Add a randomly chosen set of positions to get the seed, with multiplication values to provide more randomness. This is arbitrary and can be changed
    uint16_t seed = (*movedBoard)[0][0] + 2*((*movedBoard)[0][3]) + 3 * ((*movedBoard)[1][2]) + 5 * ((*movedBoard)[3][0]) + 7 * ((*movedBoard)[3][3]);

    srand(seed);
    uint32_t randomVal = rand();
    while((*movedBoard)[((randomVal % SIZE)/WIDTH)][((randomVal % SIZE)%HEIGHT)] != 0){
        randomVal = rand();
    }
    uint32_t randomSpace = randomVal % SIZE;
    randomVal = rand();
    uint32_t randdist = 9*(RAND_MAX/10);
    if(randomVal >= randdist){
        (*movedBoard)[randomSpace/WIDTH][randomSpace%HEIGHT] = 4;
    }
    else{
        (*movedBoard)[randomSpace/WIDTH][randomSpace%HEIGHT] = 2;
    }
}