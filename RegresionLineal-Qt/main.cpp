#include "exeigennorm.h"
#include "linealregression.h"

#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iostream>

/*En primer lugar se creara una clase llamda ExEigenNorm, la cual nos permitira
 * leer un dataset, extraer la informacion del dataset, montar sobre estructura Eigen, para normalizar los datos
 *
 */

int main(int argc, char *argv[])
{
    /*Se crea un objeto del tipo ExEigenNorm, se incluyen los 3 argumentos del contructor
     * nombre del dataset, delimitador, y nuestro flag(header o no)
     */
    ExEigenNorm extraccion(argv[1], argv[2], argv[3]);

    /* Se leen ls datos del archivo por la funcion LeerCSV*/
    std::vector<std::vector<std::string>> dataFrame = extraccion.LeerCSV();
    /*
         * Para probar la segunda función CSVtoEigen() se define la cantidad de filas y columnas
         * basados en los datos de entrada
        */
    Linealregression LR;
    int filas = dataFrame.size()+1;
    int columnas = dataFrame[0].size();
    Eigen::MatrixXd matrizDataF = extraccion.CSVtoEigen(dataFrame,filas,columnas);

    //std::cout<<matrizDataF<<std::endl;
    //std::cout << "Hello World!" << std::endl;
    Eigen::MatrixXd normMatriz = extraccion.Normalizacion(matrizDataF);
    std::cout<<"\nDatos Normalizados:\n"<<normMatriz<<std::endl;




    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(normMatriz,0.8);

        /*
         * Se desempaca la tupla, se usa std::tie
         * https://en.cppreference.com/w/cpp/utility/tuple/tie
         */

        Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;

        std::tie(X_Train,y_Train,X_Test,y_Test) = divDatos;

        /*
         * Inspección visual de la división de los datos para entrenamiento y prueba
         */
        std::cout<<"\nTamaño original\t\t\t-> "<<normMatriz.rows() <<std::endl;
        std::cout<<"*** VARIABLES INDEPENDIENTES ***"<<std::endl;
        std::cout<<"Tamaño entrenamiento(filas)\t-> "<<X_Train.rows() <<std::endl;
        std::cout<<"Tamaño entrenamiento (columnas)\t-> "<<X_Train.cols() <<std::endl;
        std::cout<<"Tamaño prueba (filas)\t\t-> "<<X_Test.rows() <<std::endl;
        std::cout<<"Tamaño prueba (columnas)\t-> "<<X_Test.cols() <<std::endl;
        std::cout<<"*** VARIABLES DEPENDIENTES ***"<<std::endl;
        std::cout<<"Tamaño Entrenamiento (filas)\t-> "<<y_Train.rows() <<std::endl;
        std::cout<<"Tamaño prueba (columnas)\t-> "<<y_Test.cols() <<std::endl;









    /*Para probar el primer algortimo lineal en donde se probara con los datps de los
     * vinos, se presentara la  regresion lineal para multiples variables, dada la naturaleza de la regresion lineal si se tiene variables con
     * diferentes unidades, una variables podria beneficiar/estropear otra variables: se necesita estandarizar los datos, dejando a todas las variables
     * del mismo orden de magnitud y contradas en  cero
     * normalizacion basada en el set score normalizacion, se necesita tres funciones: la funcion de normaliacion,
     * la del promedio y la desviacion estandar
     */





        /* A continuacion se procede a probar  la clase regresion lineal*/

        Eigen:: VectorXd vectorTrain = Eigen::VectorXd::Ones(X_Train.rows());
        Eigen:: VectorXd vectorTest = Eigen::VectorXd::Ones(X_Test.rows());

        /* Redimension de las matrices para la ubicacion en los vectores de uno ONES( con
         * referencia reshape con Numpy)*/
        X_Train.conservativeResize(X_Train.rows(),X_Train.cols() + 1);
        X_Train.col(X_Train.cols() - 1) = vectorTrain;

        X_Test.conservativeResize(X_Test.rows(),X_Test.cols() + 1);
        X_Test.col(X_Test.cols() - 1) = vectorTest;

        /* Se define el vector theta que pasara al algoritmo de gradiente descendiente (basicamente
         * un vector de ZEROS del mismo tamaño del vector de entrenamiento. Adicionalmente se pasra
         * alpha y el numero de iteraciones */

         Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_Train.cols());
         float alpha = 0.01;
         int iteraciones= 1000;

         /* Se definen la variables de salida que representan los coeficiente y el vector de costo */

         Eigen::VectorXd thetaOut;
         std::vector<float>costo;


         std::tuple<Eigen:: VectorXd,std::vector<float>> gradienteD = LR.GradienteDescendiente(X_Train,y_Train,theta,alpha,iteraciones);
         std:: tie(thetaOut,costo) = gradienteD;


          /* Se imprimen los valores de los coeficientes theta para cada FEATURES */

         std::cout<<"\n Theta: " << thetaOut <<std::endl;
         std:: cout<<"\n Costo \n "<<std::endl;
         for(auto valor:costo){

             std:: cout<<valor<<std::endl;
         }

      /* Exportamos a ficheros ,costo y thetaOut */

      extraccion.VectorToFile(costo,"Costo.txt");
      extraccion.EigenToFile(thetaOut,"\n Theta.txt");




    return EXIT_SUCCESS;
}












