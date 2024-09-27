#include <iostream>
void Rock_Papper_Scissors(const char player1_choice,const char player2_choice){
    if (player1_choice == player2_choice)
        std::cout << "It's a draw!";
    else if (player1_choice == 'r'){
        if (player2_choice == 's')
            std::cout << "Player 1 wins!";
        else if (player2_choice == 'p')
            std::cout << "Player 2 wins!";
    }
    else if (player1_choice  == 's'){
        if (player2_choice == 'r')
            std::cout << "Player 2 wins!";
        else if (player2_choice == 'p')
            std::cout << "Player 1 wins!";
    }
    else if (player1_choice == 'p'){
        if (player2_choice == 'r')
            std::cout << "Player 1 wins!";
        else if (player2_choice == 's')
            std::cout << "Player 2 wins!";
    }
}

int main(){
    char p1_choice = 'p';
    char p2_choice = 'p';
    Rock_Papper_Scissors(p1_choice, p2_choice);
}