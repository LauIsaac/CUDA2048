#include <stdio.h>
#include "types.h"
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>




void upSolver(Board * input, Board * output);
void downSolver(Board * input, Board * output);
void leftSolver(Board * input, Board * output);
void rightSolver(Board * input, Board * output);
void randGen(Board * movedBoard);
status moveHandler(Board *input, Move currMove, Board * output);

uint32_t score(Board * input);

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




    Board * out;
    out = malloc(sizeof(Board));
    status yow;
    yow = moveHandler(&Ayye,down,out);
    if(yow != boardUpdated){
        printf("This function was unable to move the board: %d \r\n",yow);
        exit(25);
    }
    else{
        for(int i =0; i < HEIGHT;i++){
            for(int j=0;j<WIDTH;j++){
                printf("[%d] ",(*out)[i][j]);
            }
            printf("\r\n");
        }
    }
    printf("Completed\r\n");
    while(1){

    }

    return 0;
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
            upSolver(input,output);
            break;
        case down:
            downSolver(input,output);
            break;
        case left:
            leftSolver(input,output);
            break;
        case right:
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
        }
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

    if(randomVal >= (9*RAND_MAX/10)){
        (*movedBoard)[randomSpace/WIDTH][randomSpace%HEIGHT] = 4;
    }
    else{
        (*movedBoard)[randomSpace/WIDTH][randomSpace%HEIGHT] = 2;
    }
}