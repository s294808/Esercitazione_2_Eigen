#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;

VectorXd SolveSystemPALU (const Matrix2d& A, const Vector2d& b, const Vector2d& exactSolution, const bool& condition)
{
    VectorXd x = A.fullPivLu().solve(b);

    double errRel;
    errRel = (exactSolution - x).norm() / exactSolution.norm();

    if(condition){
        cout << scientific << "La soluzione del sistema lineare con la decomposizione PALU e'\n " << x <<endl;
        cout << scientific << "L'errore relativo con la decomposizione PALU e' " << errRel << endl;
    }else{
        cout<< "La matrice e' singolare e il sistema lineare non e' risolvibile con la decomposizione PALU" << endl;
    }

    return x;
}

VectorXd SolveSystemQR (const Matrix2d& A, const Vector2d& b, const Vector2d& exactSolution, const bool& condition)
{
    VectorXd x = A.colPivHouseholderQr().solve(b);

    double errRel;
    errRel = (exactSolution - x).norm() / exactSolution.norm();

    if(condition){
        cout << scientific << "La soluzione del sistema lineare con la decomposizione QR e'\n " << x <<endl;
        cout << scientific << "L'errore relativo con la decomposizione QR e' " << errRel << endl;
    }else{
        cout<< "La matrice e' singolare e il sistema lineare non e' risolvibile con la decomposizione QR" << endl;
    }

    return x;
}

bool condition (const Matrix2d A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValues = svd.singularValues();

    for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) < 1e-16) {
            return false;
        }
    }

    return true;
}

int main()
{
    //RISOLVERE SISTEMI LINEARI CON LA DECOMPOSIZIONE PALU E LA DECOMPOSIZIONE QR

    Matrix2d A1;
    Vector2d b1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01,
        1.672384680188350e-01;

    Matrix2d A2;
    Vector2d b2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04,
        4.259549612877223e-04;

    Matrix2d A3;
    Vector2d b3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10,
        4.266924591433963e-10;

    Vector2d exactSolution;
    exactSolution << -1.0e+0, -1.0e+00;

    cout << scientific << "La matrice e' \n" << A1 << endl;
    cout << scientific << "Il vettore e' \n" << b1<< endl;
    SolveSystemPALU(A1, b1, exactSolution, condition(A1));
    SolveSystemQR(A1, b1, exactSolution, condition(A1));


    cout << " " << endl;

    cout << scientific << "La matrice e' \n" << A2 << endl;
    cout << scientific << "Il vettore e' \n" << b2<< endl;
    SolveSystemPALU(A2, b2, exactSolution, condition(A2));
    SolveSystemQR(A2, b2, exactSolution, condition(A2));

    cout << " " << endl;

    cout << scientific << "La matrice e' \n" << A3 << endl;
    cout << scientific << "Il vettore e' \n" << b3<< endl;
    SolveSystemPALU(A3, b3, exactSolution, condition(A3));
    SolveSystemQR(A3, b3, exactSolution, condition(A3));

    return 0;
}
