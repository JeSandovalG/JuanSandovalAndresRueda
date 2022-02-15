#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

double f(double x){
    return exp(-pow(x,2));
}

double fp(double x,double h){
    return (f(x+h)-f(x-h))/(2*h);
}

int main()
{
    double h = 0.01;
    int limite = 20/h;
    
    std::ofstream *File;
    File = new std::ofstream[2];
    File[0].open("datosDerivadaCentral.txt",std::ofstream::trunc);
    
    std::cout<< File << std::endl;
    
    for(int i = -limite; i<limite+1;i++){
        File[0]<<i*h<<" "<<fp(i*h,h)<<std::endl;
    }
    
    File[0].close();
    return 0;
}
