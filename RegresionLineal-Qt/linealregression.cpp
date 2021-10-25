#include "linealregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica minimizar alguna funcion
 * de costo, y de esta forma se puede medir la precision de la funcion de
 * hipotesis.La funcion de costo es la forma de penalizar al modelo por
 * cometer un error
 */

float Linealregression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y,Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia= pow((X * theta - y).array(),2);

    //std::cout<<"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ntheta = "<<theta<<std::endl;
    return (diferencia.sum()/(2*X.rows()));
}

/* Se implementa la funcion para dar al algoritmo los valores de theta iniciales,
 * que cambiaran iterativamente hasta que converga al valor minimo de la funcion
 * de costo. Basicamente describira el gradiente descendiente: el cual esta dado por la derivada
 * parcial de la funcion. La funcion tiene un alpha que representa el salto de gradiente y el numero
 * de iteraciones que se necesitan para actualizar theta hasta que la funcion converga minimo esperado
 */

 std:: tuple<Eigen:: VectorXd,std::vector<float>>Linealregression:: GradienteDescendiente(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd theta,float alpha,int iteraciones){
    /* Almacenamiento temporal para los valores de theta */

    Eigen::MatrixXd temporal=theta;

    /* Variable con la cantidad de parametros m(FEATURES)
     */
     int paramentros=theta.rows();

    /* Ubicar el costo inicial, que se actualizara iterativamente con los pesos */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X,y,theta));

    /* Alli por cada iteracion se calcula la funcion de error */
     for(int i = 0; i < iteraciones; ++i ){
        Eigen::MatrixXd error = X*theta - y;
        for(int j = 0; j < paramentros ; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            std::cout<<"x_i "<<X_i.cols()<<"f: "<<X_i.rows()<<std::endl;
            std::cout<<"error "<<error.cols()<<"f: "<<error.rows()<<std::endl;
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            std::cout<<"t: "<<termino.cols()<<"f: "<<termino.sum()<<std::endl;
            temporal(j,0) = theta(j,0) - ((alpha/X.rows())*termino.sum());
        }
        theta=temporal;
        costo.push_back(FuncionCosto(X,y,theta));
    }
    return  std::make_tuple(theta,costo);
}
