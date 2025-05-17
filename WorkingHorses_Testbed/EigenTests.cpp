// #include "EigenTests.h"

#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision
#include <Eigen/Dense>
#include <algorithm> // For std::copy

#include "Portfolio.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>


Eigen::MatrixXd generateAndNormalize(int m, int n, std::optional<unsigned int> seed = std::nullopt) {
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

int EigenTests::Start()
{
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
#include <iostream>

#include "EigenTests.h"

int main( int argc, char *argv[] ) 
{
    EigenTests myEigenTests;
    return  myEigenTests.Start();
}
*/
/*
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int EigenTests::Start() 
{
    //-> Performances in % (01.2015-01.205) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets =  4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)

    double performanceTable[numAssets][numQuotes] =  
           {
               {0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,   0.474,  0.6105,   0.726},
               {0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784},
               {0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965},
               {0,  0.0667,  0.1291,  0.2164,   0.148,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694}
           };

    //-> Generate a random matrix, the columns of which are weight sets
    const size_t rows = numAssets;
    const size_t cols = 10;
    const size_t seedValue = 42; // Mersenne Twister engine seed value

    // Generate with a specific seed
    Eigen::MatrixXd normalizedMatrixSeeded = generateAndNormalize( rows, cols, seedValue );
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

}  // end EigenTests::start
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Eigen::MatrixXcd EigenTests::SanitizeMatrix( const Eigen::MatrixXcd& matrix ) 
{
    // Replaces "-0.0" coefficients by "0.0"

    Eigen::MatrixXcd sanitizedMatrix = matrix;

    for ( int i = 0; i < sanitizedMatrix.rows(); ++i ) 
    {
        for ( int j = 0; j < sanitizedMatrix.cols(); ++j ) 
        {
            if ( sanitizedMatrix(i,j).real() == -0.0)  sanitizedMatrix(i,j).real( 0.0 );
            if ( sanitizedMatrix(i,j).imag() == -0.0)  sanitizedMatrix(i,j).imag( 0.0 );
        }
    }

    return sanitizedMatrix;

}  // end EigenTests::SanitizeMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void PrintMatrix( Eigen::MatrixXd mat, size_t width, size_t precision )
{
    for ( int i = 0; i < mat.rows(); i++ )
    {
        for ( int j = 0; j < mat.cols(); j++ ) 
            std::cout << std::fixed << std::setprecision(precision) << std::setw(width) << mat(i, j);
        std::cout << "\n";
    }

    std::cout << std::endl;

}  // end EigenTests::PrintMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
*/
