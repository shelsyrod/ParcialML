#ifndef LINEALREGRESSION_H
#define LINEALREGRESSION_H

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>


class Linealregression
{
public:
    Linealregression()
    {}

    float FuncionCosto(Eigen:: MatrixXd X, Eigen:: MatrixXd y,Eigen::MatrixXd theta);
    std:: tuple<Eigen:: VectorXd,std::vector<float>> GradienteDescendiente(Eigen:: MatrixXd X, Eigen:: MatrixXd y,Eigen::MatrixXd theta,float alpha,int iteraciones);

};

#endif // LINEALREGRESSION_H
