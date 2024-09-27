#include <iostream>
#include <string>
// Define student structure
struct  Student{
    std::string name;
    std::string email;
    std::string username;
    std::string experiment;
};

// Function to print out student info
void print_student_info(const Student &student){
    std::cout << student.name << std::endl;
    std::cout << student.email << std::endl;
    std::cout << student.username << std::endl;
    std::cout << student.experiment << std::endl;
}

int main(){
    // Initialize two different students
    struct Student Frank = {"Frank",
                            "fstrug@uic.edu",
                            "fstrug",
                            "CMS"};
    struct Student Alice =  {"Alice",
                            "alice231@uic.edu",
                            "aaton",
                            "ALICE"};

    // Print out students' info
    print_student_info(Frank);
    std::cout << std::endl;
    print_student_info(Alice);
}