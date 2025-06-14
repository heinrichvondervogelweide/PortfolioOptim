#include <Eigen/Dense>

#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>      // for std::pair
#include <random>
#include <functional>   // For std::function
#include <optional>

using namespace Eigen;


struct RiskInterval{ double lwb; double upb; };
struct ReturnInterval { double lwb; double upb; };
struct RiskReturnPair { double _risk; double _return; };
typedef std::vector<RiskReturnPair> RiskRetVector;


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
RiskInterval GetRiskInterval( const RiskRetVector& riskRetVec )
{
    auto minRiskPair = std::min_element( riskRetVec.begin(), riskRetVec.end(),
                                         [](const RiskReturnPair& a, const RiskReturnPair& b) 
                                         {
                                             return a._risk < b._risk;
                                         }
                                       );
    auto maxRiskPair = std::max_element( riskRetVec.begin(), riskRetVec.end(),
                                         [](const RiskReturnPair& a, const RiskReturnPair& b) 
                                         {
                                             return a._risk > b._risk;
                                         }
                                       );
    return { minRiskPair->_risk, maxRiskPair->_risk };

}  // end GetRiskInterval
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
MatrixXd& CovarianceMatrix( const MatrixXd& dataMat, MatrixXd& covMat, bool sampleCov = true )
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

    const size_t nVars = dataMat.cols();
    covMat.resize( nVars, nVars );
    covMat.setConstant( -std::numeric_limits<double>::infinity() );     // Set all coefficients to negative infinity

    VectorXd mean = dataMat.colwise().mean();                           // Calculate the mean of each column (variable)
    MatrixXd centeredData = dataMat.rowwise() - mean.transpose();       // Center the data by subtracting the mean of each column

    // Calculate the covariance matrix (for sample covariance dividing by N-1, for population covariance by N)
    covMat = (centeredData.transpose() * centeredData) / (sampleCov ? dataMat.rows() - 1 : dataMat.rows()); 

    return covMat;

}  // end CovarianceMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int main() 
{
    //===========
    // Test Case 
    //===========

    const size_t nRiskRetTuples = 1000;
    const size_t nRiskBars      =   20;

    RiskRetVector riskRetVec( nRiskRetTuples );
    const RiskInterval riskInterv = { 1.0, 3.0 };
    populateRiskRetVecTestCase( riskRetVec, riskInterv, RetIntervUpbTestCase, nRiskRetTuples );

    RiskRetVector riskRetTuplesBelowFrontier;
    riskRetTuplesBelowFrontier = 
        GetRiskReturnTuplesBelowFrontier( riskRetVec, riskRetTuplesBelowFrontier, riskInterv, nRiskBars );
    
    for ( int i = 0; i < nRiskBars; i++ )
    {
        std::cout << std::fixed << std::setprecision(6) << std::setw(9) 
                  << riskRetTuplesBelowFrontier[i]._risk << ", " 
                  << riskRetTuplesBelowFrontier[i]._return << std::endl;
    }

    /*
    // Performances in % (01.2015-01.205) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets =  4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)

    MatrixXd mat_performanceTable(numAssets, numQuotes);
    mat_performanceTable << 
               0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,  0.4740,  0.6105,  0.7260,
               0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784,
               0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965,
               0,  0.0667,  0.1291,  0.2164,  0.1480,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694;

    VectorXd vec_avgStockPerf = CalcAvgStockPerformances( mat_performanceTable, vec_avgStockPerf );
    MatrixXd mat_portfolioCov = CovarianceMatrix( mat_performanceTable.transpose(), mat_portfolioCov );

    //-> Generate random weight vectors
    const size_t nWeightVecs = 1000;
    const std::optional<size_t> seed = 4711;       // seed for random number generator (std::nullopt by default, think of Maybe in Haskell! :-) )
    MatrixXd mat_weightVec = generateWeightVecMat( numAssets, nWeightVecs, seed );

    //-> Populate the risk-return vector with random risk-return results by applying the random weight sets to the
    //   individual average stock performances and to the covariance matrix (this is the Monte-Carlo part of the optimization procedure).
    RiskRetVector riskRetVec( nWeightVecs );

    size_t i = 0;

    for ( auto& pair : riskRetVec )
    {
        pair._return = CalcAvgPortPerf( vec_avgStockPerf, mat_weightVec.col(i++) );
        pair._risk   = CalcAvgPortVola( mat_portfolioCov, mat_performanceTable );
    }

    const RiskInterval riskInterv = GetRiskInterval( riskRetVec ); // Get the risk interval of the risk-return vector

    //-> Construct an efficient frontier by fitting an upper limiting curve to the maximum return for each risk interval
    const size_t nRiskBars = 20;


	MatrixXd data(10, 4); // 10 measurements of 4 variables
	data << 
		  -0.080600,   0.026200,   0.033600,   0.066700,
		   0.160322,   0.110115,   0.162829,   0.058498,
		   0.110142,   0.138430,   0.023879,   0.077318,
		   0.016550,   0.009947,   0.003657,  -0.056232,
		   0.040286,   0.207971,   0.305643,   0.316812,
		  -0.004471,   0.222033,  -0.106164,   0.092479,
		   0.123917,   0.286734,   0.239628,   0.350045,
		   0.051880,   0.119217,   0.175341,  -0.139891,
		   0.092605,   0.091183,   0.038093,   0.199197,
		   0.071717,   0.078989,   0.328609,   0.247728 ;

    MatrixXd covMat;
    bool sampleCov = true; // Set to false for population covariance
    covMat = CovarianceMatrix( data, covMat, sampleCov );
    std::cout << "\nCovariance Matrix:\n";
    PrintMatrix(covMat, 9, 6);
    */
    return 0;
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
