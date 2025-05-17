#pragma warning(disable : 4244)

#include "Portfolio.h"
#include "EigenTests.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


#include <Eigen/Dense>
#include <random>
#include <iostream>


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Eigen::MatrixXd generateAndNormalize(int m, int n, std::optional<unsigned int> seed) {
  // Initialize a random number generator
  std::random_device rd;
  std::mt19937 gen(seed.has_value() ? std::mt19937(seed.value()) : std::mt19937(rd()));
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  // Generate the (m x n) matrix with random uniform coefficients
  Eigen::MatrixXd matrix(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix(i, j) = distrib(gen);
    }
  }

  // Normalize each column so that the column sums are 1
  for (int j = 0; j < n; ++j) {
    double colSum = matrix.col(j).sum();
    if (colSum > 0) {
      matrix.col(j) /= colSum;
    } else {
      // Handle the case where a column sum is zero (e.g., by setting all elements to 1/m)
      matrix.col(j).fill(1.0 / m);
    }
  }

  return matrix;
}

int main() {
  int rows = 5;
  int cols = 3;
  unsigned int seedValue = 42; // Example seed value

  // Generate with a specific seed
  Eigen::MatrixXd normalizedMatrixSeeded = generateAndNormalize(rows, cols, seedValue);
  std::cout << "Generated and Normalized Matrix (with seed " << seedValue << "):\n";
  std::cout << normalizedMatrixSeeded << std::endl;
  std::cout << "Column Sums:\n";
  std::cout << normalizedMatrixSeeded.colwise().sum() << std::endl;

  std::cout << "\n--------------------\n";

  // Generate without a seed (will be different each time)
  Eigen::MatrixXd normalizedMatrixNoSeed = generateAndNormalize(rows, cols);
  std::cout << "Generated and Normalized Matrix (without seed):\n";
  std::cout << normalizedMatrixNoSeed << std::endl;
  std::cout << "Column Sums:\n";
  std::cout << normalizedMatrixNoSeed.colwise().sum() << std::endl;

  return 0;
}
/*
Eigen::MatrixXd generateAndNormalize( size_t rows, size_t cols , std::optional<unsigned int> seed ) 
{
    //-> Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen( seed.has_value() ? std::mt19937(seed.value()) : std::mt19937(rd()) );
    std::uniform_real_distribution<> distrib( 0.0, 1.0 );

    auto rng = std::bind( distrib, gen );   // creates a callable object rng that generates a new random number

    //-> Generate the (m x n) matrix with random uniform coefficients
    Eigen::MatrixXd matrix( rows, cols );
    matrix.unaryExpr( [&](double dummy){ return rng(); } ); // Apply the lambda function that calls rng()) to each element of the matrix
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            double x = matrix(i, j);
        }
    }
    //-> Normalize each column so that the column sums are 1
    for ( size_t j = 0; j < cols; j++ ) 
    {
        double colSum = matrix.col(j).sum();
        matrix.col(j) = ( colSum > 0 ) ? Eigen::VectorXd(matrix.col(j)/colSum) : Eigen::VectorXd::Constant(rows, 1/rows);
    }

    return matrix;

}  // end generateAndNormalize
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
*/