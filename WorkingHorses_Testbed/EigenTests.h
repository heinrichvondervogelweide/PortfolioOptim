#pragma once

#include <Eigen/Dense>


void PrintMatrix( Eigen::MatrixXd mat, size_t width = 8, size_t precision = 2 );


class EigenTests
{
  public:
    int Start();

  private:
      Eigen::MatrixXcd SanitizeMatrix( const Eigen::MatrixXcd& matrix );

};  // end class EigenTests

