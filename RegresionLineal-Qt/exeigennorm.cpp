#include "exeigennorm.h"
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include<eigen3/Eigen/Dense>

/*Primera funcion: Lectura de ficheros csv
 * vector de vectores String
 * La idea es leer linea por linea y almacenar en un vector de vectores tipo String*/
std::vector<std::vector<std::string>> ExEigenNorm::LeerCSV(){
    /* Se abre el archivo para lectura solamente*/
    std::ifstream Archivo(setDatos);
    /* Vector de vectores del tipo string que tendra los datos del dataset*/
    std::vector<std::vector<std::string>> datosString;
    /*Se itera a traves de cada linea del dataset, y se divide el contenido
     * dado por el delimitador provisto por el contructor*/
    std::string linea="";

    while(getline(Archivo,linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /*Se cierra el archivo */
    Archivo.close();

    /*Se retorna el vector de vectores de tipo string*/
    return datosString;
}


 /* Se crea la segunda funcion para guardar el vector de vectores del tipo string
 * a una matriz Eigen. Similar a Pandas(Python) para presentar un dataFrame
 */
Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int col){
    /*
     * Si tiene cabecera la removemos
     */
    if(header==true){
        filas -= 1;
    }
    /*
     *
     * Se itera sobre filas y columnas para almacenar en la matriz vacia(Tamaño filas*columnas), que basicamente
     * almacenará string en un vector: Luego lo pasaremos a float para ser manipulados
    */
    Eigen::MatrixXd dfMatriz(col, filas);
    for(int i=0 ; i< filas ; i++){
        for(int j=0 ; j< col ; j++){
            dfMatriz(j,i) = atof(datosString[i][j].c_str());
        }
    }
    /*
     * Se transpone la matriz para tener filas por columnas
    */
    return dfMatriz.transpose();

/*A continuacion se van a implementar las funciones para la normalizacion.*/


/*En c++, la palabra clave auto especifica que el tipo de variable que se empieza a declarar
 * de deducira automaticamente de su inicializador y para las funciones,
 * si su tipo de retorno es auto, se evaluara mediante la expresion
 * del tipo de retorno en tiempo de ejecucion
 */


    /*En c++ la herencia del tipo de dato no es directa
     * o no se sabe que tipo de dato debe retornar, entonces para ello
     * para ello se declara el tipo en una expresion decltype' con el fin de tener seguridad de ue tipo de dato retorna
     */

}
    auto ExEigenNorm::Promedio(Eigen::MatrixXd datos)->decltype(datos.colwise().mean()){
            std::cout<<"Promedio: \n"<<datos.colwise().mean()<<std::endl;
            return datos.colwise().mean();
    }


/*Para implementar la funcion de desviacion estandar
 *  datos = x_1 - promedio(x)*/

    auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos)->decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt()){
        std::cout<<"\nDesviacion: \n"<<((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt()<<std::endl;
        return ((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt();
    }

    Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd datos){

        Eigen::MatrixXd diferenciaPromedio = datos.rowwise() - Promedio(datos);



        Eigen::MatrixXd matrizNormalizada = diferenciaPromedio.array().rowwise()/Desviacion(diferenciaPromedio);


        return matrizNormalizada;
    }
    /*
    Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd datos, bool siNormalizada){
        Eigen::MatrixXd dataNorm;
        if (siNormalizada==true){
            dataNorm = datos;
        }
        else{
            dataNorm = datos.leftCols(datos.cols()-1);
        }
        Eigen::MatrixXd diferenciaPromedio = dataNorm.rowwise()-Promedio(datos);
        Eigen::MatrixXd matrizNormalizada = diferenciaPromedio.array().rowwise()/Desviacion(diferenciaPromedio);

        if (siNormalizada == false){
            matrizNormalizada.conservativeResize(matrizNormalizada.rows(),matrizNormalizada.cols()+1);
            matrizNormalizada.col(matrizNormalizada.cols()-1) = datos.rightCols(1);

        }

        return matrizNormalizada;
    }
*/


    /*
     * A continuación se hará una función para dividir los datos en conjunto de datos de entrenamiento
     * y conjunto de datos de prueba
     */
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> ExEigenNorm::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain){
        int filas = datos.rows();
        int filasTrain = round(sizeTrain * datos.rows());
        int filasTest = datos.rows() - filasTrain;

        /*
         * Con Eigen se puede especificar el bloque de una matriz, por ejemplo se pueden seleccionar
         * las filas superiores para el conjunto de datos de entrenamiento indicando cuantas filas
         * se desean, se seleccionan desde 0 (fila 0) hasta el número de filas indicado
         */

        Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);

        /*
         * Seleccionadas las filas superiores para entrenamiento, se seleccionan las 11 primeras
         * columnas (columnas a la izquierda) que representan las variables independientes (Features)
         */

        Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);

        /*
         * Se selecciona ahora la variable dependiente que corresponde a la ultima columna
         */

        Eigen::MatrixXd y_train = entrenamiento.rightCols(1);

        /*
         * Se realiza lo mismo para el conjunto de pruebas
         */

        Eigen::MatrixXd test = datos.bottomRows(filasTest);
        Eigen::MatrixXd X_test = entrenamiento.leftCols(datos.cols()-1);
        Eigen::MatrixXd y_test = entrenamiento.rightCols(1);

        /*
         * Finalmente se retorna la tupla dada por el conjunto de datos de prueba y entrenamiento
         */

        return std::make_tuple(X_train,y_train,X_test,y_test);
    }

    void ExEigenNorm::VectorToFile(std:: vector<float> vector,std::string nombre){
            std::ofstream fichero(nombre);
            std::ostream_iterator<float> iterador(fichero,"\n");
            std::copy(vector.begin(),vector.end(),iterador);
    }

    void ExEigenNorm:: EigenToFile(Eigen::MatrixXd datos,std::string nombre){
        std::ofstream fichero(nombre);
        if(fichero.is_open()){
            fichero<<datos<<"\n";
        }
    }
