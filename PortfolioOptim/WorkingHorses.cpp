#include "WorkingHorses.h"

#include <iostream>
#include <iomanip>


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <random>

RiskRetVector& populateRiskRetVecTestCase( RiskRetVector& riskRetVec, 
                                           RiskInterval riskInterv, 
                                           std::function<double(double)> retIntervUpbFunc,
                                           size_t nTuples,
                                           std::optional<unsigned int> seed                )
{
    // Generates nTuples of random risk-return tuples (x,y) with x within a given
    // risk interval and y within the interval [0, retIntervUpbFunc(x)].
    // 
    // Input:
    // ======
    // riskInterv
    //      Risk interval [lwb, upb] defined by lower and upper bound.
    // 
    // retIntervUpbFunc
    //      Upper bound function for the return interval, which is a function of the risk value.
    // 
    // nTuples
    //      Requested number of tuples to be generated. Will be the return size of riskRetVec.
    // 
    // seed:
    //      Optional input parameter of type <unsigned int>; if provided random number generator
    //      will be seeded with this value. If not provided, a random seed will be used.
    //
    // Output and Return: 
    // ==================
    // 
    // riskRetVec
    //      Vector of RiskReturn tuples are stored in Pair structs.

    std::random_device rd;  // Generator for a random size_t number
    std::mt19937 gen( seed.has_value() ? std::mt19937(seed.value()) : std::mt19937(rd()) );
    std::uniform_real_distribution<> riskDistrib( riskInterv.lwb, riskInterv.upb );

    for ( size_t i = 0; i < nTuples; i++ )
    {
        const double risk = riskDistrib( gen );
        const ReturnInterval retInterv = { 0, retIntervUpbFunc(risk) }; 
        std::uniform_real_distribution<> retDistrib( retInterv.lwb, retInterv.upb );
        double ret = retDistrib( gen );
        riskRetVec.push_back( {risk, ret} );
    }

    return riskRetVec;

}  // end populateRiskRetVecTestCase
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>

Eigen::VectorXd& CalcAvgStockPerformances( const Eigen::MatrixXd& perfMat, 
                                           Eigen::VectorXd& stockPerformances )   
{
    // Calculate the average annual performance of each stock in the portfolio.
    //
    // Input: 
    // ======
    // 
    // perfMat
    //    Performance matrix of stocks, with the performance given as a percentage
    //    over time with each stock starting at 0%. The rows are the values for a 
    //    given stock, and the columns represent points in time.
    //
    // Output and Return: 
    // ==================
    // 
    // stockPerformances
    //    Vector containing the average performance of each stock in the portfolio.
    //
    // The average performances are calculated by fitting a linear regression line
    // to the log of the normalized performances (starting with 100% for each stock)
    // through the origin [log(100%)=0 at t = 0]. 
    // The slope of this line is assumed to be a measure for the average performance.

    const size_t numStocks = perfMat.rows();
    const size_t numClosPrice = perfMat.cols();
    stockPerformances.resize( numStocks, 1 );

    Eigen::MatrixXd logPerfMat = (perfMat.array() + 1).log();	// Take natural log of normalized performance (starting with 100%)
    
    // Create vector containing numbers from 0 up to (but not including) n
    Eigen::VectorXd X(numClosPrice);
    X = Eigen::VectorXd::LinSpaced( numClosPrice, 0.0, numClosPrice - 1.0 );  // n elements, from 0 to n equally spaced
    
    Eigen::VectorXd logSlope( numStocks, 1 );

    // Linear regression using matrix algebra: slope = (X'X)^(-1) * X'Y
    logSlope = ( X.transpose() * X ).inverse() * X.transpose() * logPerfMat.transpose();
    stockPerformances = logSlope.array().exp() - 1;

    return stockPerformances;

}  // CalcAvgStockPerformances
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>

double CalcAvgPortPerf( const Eigen::VectorXd& stockPerfAvg, const Eigen::VectorXd& weights )
{
    // Calculate the average performance of a portfolio given the average performance 
    // of each stock and the weights of each stock in the portfolio.

    const size_t numStocks = stockPerfAvg.rows();

    if ( weights.rows() != numStocks ) throw std::invalid_argument( "weights.size() != stockPerfAvg.size()" );
    if ( weights.sum() - 1 > 1e-6 )    throw std::invalid_argument( "weights.sum() != 1" ); 
    
    return stockPerfAvg.dot( weights ); // Calculate the dot product

}  // CalcAvgPortPerf
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
MatrixXd& CalcSimpleRelRetMatrix( const MatrixXd& perfMat, MatrixXd& relRetMat )
{
    // Calculate the simple relative returns of a performance matrix.
    // The relative return is defined as the price ratio minus 1, 
    //     P(j+1)/P(j) - 1
    // or if expressed by the normalized zero based percentage performance p(j) as
    //     ( p(j+1) - p(j)) ) / ( 1 + p(j) )
    //
    // Input: 
    // ======
    // 
    // perfMat
    //    Performance matrix of stocks, with the performance given as a percentage
    //    over time with each stock starting at 0%. The rows are the values for a 
    //    given stock, and the columns represent points in time.
    //
    // Output and Return: 
    // ==================
    // 
    // relRetMat
    //    Matrix containing the simple relative returns for each historical time step
    //    and for each stock in the portfolio as a percentage increase per time step.

    const size_t numStocks = perfMat.rows();
    const size_t numClosPrice = perfMat.cols();

    relRetMat.resize(numStocks, numClosPrice - 1);  // number of columns is one less than that of the performance matrix
    relRetMat.setZero();                            // Initialize all elements to zero

    //-> Arrays of the leftmost and rightmost columns of the performance matrix
    const Eigen::ArrayXXd& leftPerfArray  = perfMat.leftCols(numClosPrice-1).array();  
    const Eigen::ArrayXXd& rightPerfArray = perfMat.rightCols(numClosPrice-1).array(); 

    //-> Calculate the simple return ratio matrix
    relRetMat = ( rightPerfArray - leftPerfArray ) / ( 1 + leftPerfArray);   // ( p(j+1) - p(j)) ) / ( 1 + p(j) )

    return relRetMat;

}  // end CalcSimpleRelRetMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
MatrixXd& CalcLogRetMatrix( const MatrixXd& perfMat, MatrixXd& logRetMat )
{
    // Calculate the differences of the logarithms of prices, where normalized 
    // prices P(i,j) are calculated from the normalized zero-based percentage
    // performance p(i,j) by 
    //                          P(i,j) = 1 + p(i,j)
    //
    // Input: 
    // ======
    // 
    // perfMat
    //    Performance matrix of stocks, with the performance given as a percentage
    //    over time with each stock starting at 0%. The rows are the values for a 
    //    given stock, and the columns represent points in time.
    //
    // Output and Return: 
    // ==================
    // 
    // logRetMat
    //    Matrix containing the log of the price ratios for each historical time step
    //    and for each stock in the portfolio.
    // 
    //    Keep in mind the following relation between relative returns and log returns:
    //    integral( dP(t)/P(t), P(j), P(j+1), t ) = ln(P(j+1)) - ln(P(j)) = ln( P(j+1) / P(j) )

    const size_t numStocks = perfMat.rows();
    const size_t numClosPrice = perfMat.cols();

    logRetMat.resize( numStocks, numClosPrice-1 );
    logRetMat.setZero();  // Initialize all elements to zero

    //-> Arrays with normalized prices for the leftmost and rightmost columns of the performance matrix
    const Eigen::ArrayXXd& leftPerfArray  = 1 + perfMat.leftCols(numClosPrice-1).array();  
    const Eigen::ArrayXXd& rightPerfArray = 1 + perfMat.rightCols(numClosPrice-1).array(); 

    //-> Take the log of the ratio for the return matrix
    logRetMat = ( rightPerfArray / leftPerfArray ).log();   // natural logarithm

    return logRetMat;

}  // end CalcLogRetMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
MatrixXd& CalcCovarianceMatrix( const MatrixXd& returnsMat, MatrixXd& covMat, bool sampleCov )
{
    //-> Calculates the covariance matrix of the given data matrix.
    // 
    //   The data matrix is a matrix, where the columns represent variables 
    //   and the rows measurements of these variables.
    // 
    //   The returned covariance matrix is a square matrix the size of which is 
    //   determined by the number of variables.
    //  
    //   Parameter sampleCov determines whether the covariance is calculated 
    //   for the sample or for the underlying population.

    const size_t nVars = returnsMat.cols();
    covMat.resize( nVars, nVars );
    covMat.setConstant( -std::numeric_limits<double>::infinity() );     // Set all coefficients to negative infinity

    VectorXd mean = returnsMat.colwise().mean();                           // Calculate the mean of each column (variable)
    MatrixXd centeredData = returnsMat.rowwise() - mean.transpose();       // Center the data by subtracting the mean of each column

    // Calculate the covariance matrix (for sample covariance dividing by N-1, for population covariance by N)
    covMat = (centeredData.transpose() * centeredData) / (sampleCov ? returnsMat.rows() - 1 : returnsMat.rows()); 

    return covMat;

}  // end CovarianceMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Eigen::MatrixXd getCorrMatrix( const Eigen::MatrixXd& covMat, Eigen::MatrixXd& corrMat )
{
    // Calculates the correlation matrix from the covariance matrix.
    //
    // Input:               covMat  # Covariance matrix of the variables.
    // Output and Return:   corrMat # Correlation matrix of the variables.

    const size_t nVars = covMat.rows();
    corrMat.resize( nVars, nVars );
 
    // Build D^(-1), the inverse of the diagonal matrix of standard deviations
    MatrixXd stdDevInvMat = covMat.diagonal().cwiseSqrt().cwiseInverse().asDiagonal();

    // Compute correlation matrix: R = D^{-1} * Σ * D^{-1}
    corrMat = stdDevInvMat * covMat * stdDevInvMat;

    return corrMat;

}  // end getCorrMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <optional>
#include <random>
#include <Eigen/Dense>

Eigen::MatrixXd generateWeightVecMat( size_t rows, size_t cols, std::optional<unsigned int> seed ) 
{
    //-> Generate a (rows x cols) matrix with random uniform coefficients in the range [0, 1],
    //   the column sums of which are normalized to 1. The columns can be interpreted as weight vectors.
    
    //-> Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen( seed.has_value() ? std::mt19937(seed.value()) : std::mt19937(rd()) );
    std::uniform_real_distribution<> distrib( 0.0, 1.0 );

    //-> Generate the (m x n) matrix with random uniform coefficients
    Eigen::MatrixXd matrix( rows, cols );
    for ( int i = 0; i < rows; i++ ) for ( int j = 0; j < cols; j++ ) matrix(i, j) = distrib( gen );
   
    //-> Normalize each column so that the column sums are 1
    for ( size_t j = 0; j < cols; j++ ) 
    {
        double colSum = matrix.col(j).sum();
        matrix.col(j) = ( colSum > 0 ) ? Eigen::VectorXd(matrix.col(j)/colSum) : Eigen::VectorXd::Constant(rows, 1/int(rows));
    }

    return matrix;

}  // end generateWeightVecMat
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Recursive helper to generate all compositions of k into m non-negative integers
void generate_compositions( size_t k, size_t m, std::vector<size_t>& current, std::vector<Eigen::VectorXd>& results ) 
{
    if  ( m == 1 ) 
    {
        current.push_back(k);
        Eigen::VectorXd vec(current.size());
        for ( size_t i = 0; i < current.size(); ++i )  vec[i] = static_cast<double>(current[i]);
        vec /= vec.sum();       // Normalize 
        results.push_back( vec );
        current.pop_back();
        return;
    }

    for ( size_t i = 0; i <= k; ++i ) 
    {
        current.push_back(i);
        generate_compositions( k - i, m - 1, current, results );
        current.pop_back();
    }
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
std::vector<Eigen::VectorXd> generate_simplex_grid( size_t mAssets, size_t kResolution ) 
{
    std::vector<Eigen::VectorXd> results;
    std::vector<size_t> current;
    generate_compositions( kResolution, mAssets, current, results );
    return results;
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double CalcAvgPortVola(const Eigen::MatrixXd& covMat, const Eigen::VectorXd& weights)
{
    // Calculate the average portfolio volatility given the covariance matrix and the weights of each stock in the portfolio.
    //
    // Input: 
    // ======
    // 
    // covMat
    //    Covariance matrix of the stocks in the portfolio.
    //
    // weights
    //    Vector containing the weights of each stock in the portfolio.
    //
    // Return: 
    // =======
    
    if ( covMat.rows() != covMat.cols() ) throw std::invalid_argument( "covMat must be square" );
    if ( covMat.rows() != weights.size()) throw std::invalid_argument( "covMat size does not match weights size" );

    // Calculate the square root of the quadratic form
    return std::sqrt( (weights.transpose() * covMat * weights)(0, 0) );  // (0, 0) extracts the scalar from the 1×1 Eigen matrix

}  // end CalcAvgPortVola
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
RiskRetVector& GetRiskReturnTuplesBelowFrontier
(
    RiskRetVector& riskRetVec,
    RiskRetVector& riskRetTuplesBelowFrontier,
    const RiskInterval riskInterv,
    const size_t nRiskBars
)
{
    // Sort riskRetVec based on the _risk member of the RiskReturnPairs
    std::sort( riskRetVec.begin(), riskRetVec.end(), 
               [](const RiskReturnPair& a, const RiskReturnPair& b) { return a._risk < b._risk; }
             );

    const double riskBarWidth = ( riskInterv.upb - riskInterv.lwb ) / nRiskBars;

    //-> Create a vector to store the maximum return for each subinterval
    //   and initialize all elements to negative infinity to ensure afterwards initial value has been modified
    std::vector<double> maxReturn( nRiskBars, -std::numeric_limits<double>::infinity());

    //-> Iterate through your riskRetVec and assign each pair to a subinterval
    //
    for ( const auto& pair : riskRetVec ) 
    {
        if ( pair._risk >= riskInterv.lwb  &&  pair._risk <= riskInterv.upb) 
        {
            // Determine which subinterval this risk value belongs to
            int intervalIndex = static_cast<int>( (pair._risk - riskInterv.lwb) / riskBarWidth );

            // Handle the case where risk is exactly upb (to avoid going out of bounds)
            if ( intervalIndex == nRiskBars )  intervalIndex--;

            // Update the maximum return for that subinterval if the current return is higher
            if ( intervalIndex >= 0  &&  intervalIndex < nRiskBars ) 
            {
                maxReturn[intervalIndex] = std::max( maxReturn[intervalIndex], pair._return );
            }
        }
    }

    for ( int i = 0; i < nRiskBars; i++ )
    {
        const double riskBarCenter = riskInterv.lwb + (i + 0.5)*riskBarWidth;
        RiskReturnPair riskRetPair;
        riskRetPair._risk   = riskBarCenter; 
        riskRetPair._return = maxReturn[i];
        riskRetTuplesBelowFrontier.push_back( riskRetPair );
    }

    return riskRetTuplesBelowFrontier;

} // end GetRiskReturnTuplesBelowFrontier
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>

Eigen::MatrixXcd SanitizeMatrix( const Eigen::MatrixXcd& matrix ) 
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

}  // end SanitizeMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <complex>
#include <optional>
#include <Eigen/Dense>

Eigen::MatrixXcd SanitizeUsingLambda( const Eigen::MatrixXcd& matrix ) 
{
    using Complex = std::complex<double>;

    auto sanitizedMat = matrix.unaryExpr(  [](const Complex& z) 
                                           {
                                               auto fix_zero = [](double x) { return (x == -0.0 ? 0.0 : x); };
                                               return Complex( fix_zero(z.real()), fix_zero(z.imag()) );
                                           }
                                        );
    return sanitizedMat;
    
}  // end SanitizeUsingLambda
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
const Range<size_t, 0, 100> PercentageOfGridPointsUsedToDetermineTheMinMaxWeightRange( 10 );
const size_t                SimplexGridResolutionLwb = 4;           // Lower bound for the simplex grid resolution
const size_t                MaxIterations = 100;                    // Maximum number of iterations for the optimization process
const double                MaxAllowedTopWeightSpread = 1.0E-4;     // Minimum spread the top weights in the optimization process
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <sstream>
#include <string>
#include <iostream>

void PortfolioOptimizer::initialize( Eigen::VectorXd avgPerfVec, Eigen::MatrixXd portfolioCov )
{
    if (avgPerfVec.size() != myNumAssets)
        throw std::invalid_argument( "avgPerfVec.size() != myNumAssets" );
    if (portfolioCov.rows() != myNumAssets || portfolioCov.cols() != myNumAssets)
        throw std::invalid_argument( "portfolioCov must be a square matrix of size myNumAssets" );
    if (portfolioCov.hasNaN())
        throw std::invalid_argument(" portfolioCov contains NaN values" );

    myAvgPerfVec   = avgPerfVec;
    myPortfolioCov = portfolioCov;

    //-> Initialize the simplex grid, create Sharpe Ratio Rank Table,
    //   and populate the "Driving Weight Section" of the rank table with weight vectors
    //
	if ( myMinTotalNumOfGridPoints == 0 )  
	{
		mySimplexGridResolution = SimplexGridResolutionLwb;
	}
	else
	{
        mySimplexGridResolution = find_k( myNumAssets, myMinTotalNumOfGridPoints ); // Find k such that BinCoeff(k + mAssets - 1, k) == nMinGridPoints
        if (mySimplexGridResolution < SimplexGridResolutionLwb)  mySimplexGridResolution = SimplexGridResolutionLwb; // Ensure the resolution is not below the lower bound
    }

    long double n = boost::math::binomial_coefficient<long double>( mySimplexGridResolution + myNumAssets - 1, myNumAssets - 1 );
    myNumWeightVectors = static_cast<uint128_t>( n );
    if ( myNumWeightVectors > std::numeric_limits<std::size_t>::max() )  throw std::overflow_error( "size_t overflow" );

    std::ostringstream oss;
    oss << "Rank Table for " << myNumAssets << " assets with simplex grid resolution k = " << mySimplexGridResolution 
        << " and " << myNumWeightVectors << " weight vectors";
    const std::string tableName = oss.str();

    p_myRankTable = new SharpeRankTable( size_t(n), myNumAssets, tableName );
    Eigen::Block<Eigen::MatrixXd> weights = p_myRankTable->block_DrivingWeights();
    weights.setZero();   // Initialize the driving weights block to zero

    std::vector<Eigen::VectorXd> weight_vectors = generate_simplex_grid( myNumAssets, mySimplexGridResolution );

    size_t i = 0;
    for ( const auto& vec : weight_vectors )  weights.row(i++) = vec;  // Copy the generated weight vectors into the driving weights block

}  // end PortfolioOptimizer::initialize
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include "PortOptTests.h"

void PortfolioOptimizer::iterate()
{
    size_t nIterations = 0;

    while ( ++nIterations <= MaxIterations )
    {
        //-> Calculate the average performance, risk, and Sharpe Ratio for each weight vector in the rank table
        CalcPerformance();

        //-> Sort the table by Sharpe Ratio in descending order
        p_myRankTable->sort();  

        //-> For the upper subset of weights in the sorted table, 
        //   determine the minimum and maximum weights for each asset
        getMinMaxWeights();

        for ( size_t i = 0; i < myNumAssets; i++ )  std::cout << std::setw(7) << std::fixed << std::setprecision(2) << myMinWeight[i]*100;
        std::cout << "\n";
        for ( size_t i = 0; i < myNumAssets; i++ )  std::cout << std::setw(7) << std::fixed << std::setprecision(2) << myMaxWeight[i]*100;
        std::cout << "\n\n";

        if ( topWeightSpread() < MaxAllowedTopWeightSpread )  break;  // Convergence achieved, stop the iteration

        //-> Downscale the weight ranges for the individual assets to the range
        //   covered by the top nTop weight vectors
        downscaleWeightRanges();

        //-> Normalize the downscaled weights so that they sum up to 1
        normalizeWeights();

        //-> Take the normalized weights and copy them into the driving weights block
        const Eigen::Block<Eigen::MatrixXd> normWeightMat = p_myRankTable->block_NormalizedWeights();
              Eigen::Block<Eigen::MatrixXd> driveWeightMat  = p_myRankTable->block_DrivingWeights();
        driveWeightMat = normWeightMat;  // Copy the normalized weights back to the driving weights block

    }  // end while

    const Eigen::MatrixXd table = p_myRankTable->getTable();
    PrintMatrix( 100*table.block(0,0,35,4), 9, 2 );

}  // end PortfolioOptimizer::iterate
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include "PortOptTests.h"

void PortfolioOptimizer::CalcPerformance()
{
    if ( myNumWeightVectors > std::numeric_limits<std::size_t>::max() )  throw std::overflow_error( "size_t overflow" );
    size_t numWeightVectors = size_t( myNumWeightVectors );

    const Eigen::Block<Eigen::MatrixXd> weightMat = p_myRankTable->block_DrivingWeights();

    Eigen::Block<Eigen::MatrixXd> returnVec = p_myRankTable->col_Return();
    Eigen::Block<Eigen::MatrixXd> riskVec   = p_myRankTable->col_Risk  ();
    Eigen::Block<Eigen::MatrixXd> sharpeVec = p_myRankTable->col_Sharpe();

    for ( int i = 0; i < numWeightVectors; i++ ) 
    {
        Eigen::VectorXd weights = weightMat.row(i); 
        returnVec(i, 0) = CalcAvgPortPerf( myAvgPerfVec, weights );
        riskVec  (i, 0) = CalcAvgPortVola( myPortfolioCov, weights );
        sharpeVec(i, 0) = returnVec(i, 0) / riskVec(i, 0);  
    }

}  // end PortfolioOptimizer::CalcPerformance
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void PortfolioOptimizer::getMinMaxWeights()
{
    size_t n    = size_t( myNumWeightVectors );
    size_t nTop = size_t( n * PercentageOfGridPointsUsedToDetermineTheMinMaxWeightRange/100.0 ) + 1;

    const Eigen::Block<Eigen::MatrixXd> weightMat = p_myRankTable->block_DrivingWeights();

    for ( size_t i = 0; i < myNumAssets; i++ )
    {
        myMinWeight[i] = weightMat.col(i).head(nTop).minCoeff();
        myMaxWeight[i] = weightMat.col(i).head(nTop).maxCoeff();         
    }

}  // end PortfolioOptimizer::getMinMaxWeights
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void PortfolioOptimizer::downscaleWeightRanges()
{
    //-> Downscale the weight ranges for the individual assets to the range
    //   covered by the top nTop weight vectors
    const Eigen::Block<Eigen::MatrixXd> driveWeightMat  = p_myRankTable->block_DrivingWeights();
          Eigen::Block<Eigen::MatrixXd> dscaleWeightMat = p_myRankTable->block_DownscaledWeights();

    for ( size_t j = 0; j < myNumAssets; j++ )
    {
        // Zoom into the weight range covered by the topmost weight sets for each asset in the sorted table
        dscaleWeightMat.col(j) = driveWeightMat.col(j) * (myMaxWeight[j] - myMinWeight[j]);
        dscaleWeightMat.col(j).array() += myMinWeight[j];  // Shift the downscaled weight range to the minimum weight
    }

}  // end PortfolioOptimizer::downscaleWeightRanges
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void PortfolioOptimizer::normalizeWeights()
{
    const Eigen::Block<Eigen::MatrixXd> dscaleWeightMat = p_myRankTable->block_DownscaledWeights();
    Eigen::Block<Eigen::MatrixXd> normWeightMat = p_myRankTable->block_NormalizedWeights();
    Eigen::VectorXd dscaleRowSums = dscaleWeightMat.rowwise().sum();

    normWeightMat = dscaleWeightMat.array().colwise() / dscaleRowSums.array();

}  // end PortfolioOptimizer::normalizeWeights
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double PortfolioOptimizer::topWeightSpread()
{
    //-> Calculate the Euclidian distance between minima and maxima of the top weights

    Eigen::VectorXd vMin = Eigen::Map<const Eigen::VectorXd>(myMinWeight.data(), myMinWeight.size());
    Eigen::VectorXd vMax = Eigen::Map<const Eigen::VectorXd>(myMaxWeight.data(), myMaxWeight.size());

    return (vMax - vMin).norm();  // Euclidean (L2) norm

}  // end PortfolioOptimizer::topWeightSpread
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iostream>

void SharpeRankTable::sort()
{
    // Sort the table by Sharpe Ratio in descending order   
    int sortCol = getSharpeRatioColIdx();

    // Step 1: Create an std::vector of row indices
    std::vector<size_t> indices( myNumWeightGridPoints );
    std::iota( indices.begin(), indices.end(), 0 );   // std::iota fills the indices with [0, 1, ..., n-1].

    // Step 2: Sort indices based on column 3 values (descending)
    std::sort( indices.begin(), indices.end(), [&](size_t i1, size_t i2) 
                                               {
                                                   return myTable(i1, sortCol) > myTable(i2, sortCol);  // descending
                                               }
             );

    // Step 3: Create a new sorted matrix
    Eigen::MatrixXd sortedTable( myTable.rows(), myTable.cols() );

    for ( int i = 0; i < indices.size(); i++ ) 
    {
        sortedTable.row(i) = myTable.row( indices[i] );
    }

    // Optional: Replace original
    myTable = sortedTable;

}  // end SharpeRankTable::sort
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <boost/multiprecision/cpp_int.hpp>
#include <limits>
#include <stdexcept>  // for std::overflow_error

size_t safe_uint128_to_size_t( boost::multiprecision::uint128_t n ) 
{
    if ( n > std::numeric_limits<size_t>::max() ) 
    {
        throw std::overflow_error("uint128_t value exceeds size_t range");
    }

    return static_cast<size_t>(n);

}  // end safe_uint128_to_size_t
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <cstdint>
#include <boost/math/special_functions/binomial.hpp>

#include <boost/multiprecision/cpp_int.hpp>
using boost::multiprecision::uint128_t;

size_t PortfolioOptimizer::find_k( size_t mAssets, uint128_t nMinGridPoints ) 
{
    //-> Find the value of k which produces at least nMinGridPoints simplex grid points
    //   for the given number of assets mAssets.

    size_t k = 1;

    //-> Deal with special cases, which can be solved effortlessly
    //
    if ( mAssets == 1 )  return (k = 1);

    if ( mAssets == 2 )  // leads to BinCoeff(k + 1, k)
    {
        k = safe_uint128_to_size_t( nMinGridPoints - 1 );
        return k;
    }

    uint128_t nHigh;

    //-> Increase k exponentially, until n > nMinGridPoints
    //
    for ( size_t i = 0 ; ; i++ )
    {
        long double nFloatHigh = boost::math::binomial_coefficient<long double>( k + mAssets - 1, k );
        uint128_t nHigh = static_cast<uint128_t>(nFloatHigh);  
        if ( nHigh >= nMinGridPoints )  break;  // found k such that BinCoeff(k + mAssets - 1, k) >= nMinGridPoints
        k *= 2;
    }

    //-> Continue with binary search for k 
    //
    size_t kLow = 0, kHigh = k;

    for ( ; ; )
    {
        size_t kMid = ( kLow + kHigh ) / 2;
        long double nFloatMid = boost::math::binomial_coefficient<long double>( kMid + mAssets - 1, kMid );
        uint128_t nMid = static_cast<uint128_t>( nFloatMid );

        if ( nMid < nMinGridPoints )  kLow = kMid + 1; else kHigh = kMid - 1; 

        if ( kLow > kHigh )
        {
            // Found the k such that BinCoeff(k + mAssets - 1, k) <= nMinGridPoints <= BinCoeff(k-1 + mAssets - 1, k-1)
            k = kLow; 
            break;
        }
    }

    return k;

}  // end PortfolioOptimizer::find_k
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
