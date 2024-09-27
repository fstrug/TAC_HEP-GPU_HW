#include <iostream>
void switch_var(int &a,int &b){
    int temp_var = a;
    a = b;
    b = temp_var;
}
    
int main(){
    // initialize arrays
    int a[10] = {0,1,2,3,4,5,6,7,8,9};
    int b[10] = {10,11,12,13,14,15,16,17,18,19};
        
    // swap values element by element
    for (int i=0; i < 10; i++){
        switch_var(a[i], b[i]);
    }
        
    // iterate through elements of a and print them out
    for (int i=0; i < 10; i++){
        std::cout << a[i] <<",";
    }
    std::cout << "\n";
        
    // iterate through elements of b and print them out
    for (int i=0; i < 10; i++){
        std::cout << b[i] << ",";
    }
    std::cout << "\n";
}