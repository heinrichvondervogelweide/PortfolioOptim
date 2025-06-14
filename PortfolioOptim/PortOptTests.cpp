#include "WorkingHorses.h"
#include "Weights.h"


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

void PrintMatrix( Eigen::MatrixXd mat, size_t width, size_t precision )
{
    for ( int i = 0; i < mat.rows(); i++ )
    {
        for ( int j = 0; j < mat.cols(); j++ ) 
            std::cout << std::fixed << std::setprecision(precision) << std::setw(width) << mat(i, j);
        std::cout << "\n";
    }

    std::cout << std::endl;

}  // end PrintMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <iomanip>

int test_SanitizeMatrix() 
{
    using namespace Eigen;
    using namespace std;

    using Complex = complex<double>;

    MatrixXcd mat(2,2);
    mat << Complex(-0.0,  0.0), Complex(1.0, -0.0),
           Complex(-0.0, -0.0), Complex(3.0,  4.0);
    cout << "Original matrix:\n" << mat << endl;

    mat = SanitizeMatrix( mat );
    cout << "\nSanitized matrix (using SanitizeMatrix function):\n" << mat << endl;

    mat << Complex(-0.0,  0.0), Complex(1.0, -0.0),
           Complex(-0.0, -0.0), Complex(3.0,  4.0);  // reset matrix to original values

    mat = SanitizeUsingLambda( mat );
    cout << "\nSanitized matrix (using SanitizeUsingLambda function):\n" << mat << endl;

    return 0;

} // end test_SanitizeMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>
#include <iostream>

int test_CalculateAvgPortfolioPerf() 
{
    // Performances in % (01.2015-01.2025) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets =  4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)

    Eigen::MatrixXd mat_performanceTable(numAssets, numQuotes);
    mat_performanceTable << 
               0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,  0.4740,  0.6105,  0.7260,
               0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784,
               0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965,
               0,  0.0667,  0.1291,  0.2164,  0.1480,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694;

    // Calculate average performance per stock
    Eigen::VectorXd avgPerfVec;
    avgPerfVec = CalcAvgStockPerformances( mat_performanceTable, avgPerfVec );

    std::cout << "Average performances per stock:\n";
    std::cout << "Storm Fund II          :" << std::setw(7) << std::fixed << std::setprecision(2) << avgPerfVec(0) * 100 << "%\n";
    std::cout << "Quantex Global Value   :" << std::setw(7) << std::fixed << std::setprecision(2) << avgPerfVec(1) * 100 << "%\n";
    std::cout << "Polar Capital Insurance:" << std::setw(7) << std::fixed << std::setprecision(2) << avgPerfVec(2) * 100 << "%\n";
    std::cout << "UniGlobal              :" << std::setw(7) << std::fixed << std::setprecision(2) << avgPerfVec(3) * 100 << "%\n";
    std::cout << "\nCompare results with calculations in\n    $(SolutionDir)\\TestTools\\PortfolioOptim_SlnTest.xlsm\n\n";

    /*
    * Results here:
    *   Storm Fund II          :   5.05%
    *   Quantex Global Value   :  12.35%
    *   Polar Capital Insurance:   9.26%
    *   UniGlobal              :   9.63%
    * 
    * are different from those in Colab Notebook PyPortfolioOpt_FirstSteps.ipynb
    *   Average Annual Returns:
    *   Storm Fnd:              05.6098%
    *   Quantex  :              12.6072%
    *   Polar    :              11.2211%
    *   UniGlobal:              11.1166%
    * 
    * because in the latter case the average performance is calculated as the geometric mean
    * GeoMean(r(i)+1)-1 from the simple relative returns r(i) (as calculated e.g. by CalcSimpleRelRetMatrix),
    */

    // Calculate the average performance of the portfolio
    Eigen::VectorXd weights(numAssets);
    weights << 0.113636364,	0.393939394, 0.189393939, 0.303030303;

    double avgPortPerf = CalcAvgPortPerf( avgPerfVec, weights );

    std::cout << "Average performance applying the following weights:\n";
    std::cout << weights.transpose() << "\n";
    std::cout << "Average performance of the portfolio = " << std::fixed << std::setprecision(2) << avgPortPerf * 100 << "%\n";

    return 0;

} // end test_CalculateAvgPortfolioPerf
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int test_CalcRelRetMatrices()
{
    // Performances in % (01.2015-01.2025) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets =  4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)

    Eigen::MatrixXd mat_performanceTable(numAssets, numQuotes);
    mat_performanceTable << 
               0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,  0.4740,  0.6105,  0.7260,
               0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784,
               0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965,
               0,  0.0667,  0.1291,  0.2164,  0.1480,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694;

    // Calculate simple relative returns per stock
    Eigen::MatrixXd simpleRelRet;
    simpleRelRet = CalcSimpleRelRetMatrix( mat_performanceTable, simpleRelRet );

    std::cout << "Simple relative returns per stock:\n";
    std::cout << "Storm Fund II          :" << std::setw(7) << std::fixed << std::setprecision(2) << simpleRelRet.row(0) * 100 << "%\n";
    std::cout << "Quantex Global Value   :" << std::setw(7) << std::fixed << std::setprecision(2) << simpleRelRet.row(1) * 100 << "%\n";
    std::cout << "Polar Capital Insurance:" << std::setw(7) << std::fixed << std::setprecision(2) << simpleRelRet.row(2) * 100 << "%\n";
    std::cout << "UniGlobal              :" << std::setw(7) << std::fixed << std::setprecision(2) << simpleRelRet.row(3) * 100 << "%\n";
    std::cout << "\nCompare results with calculations in\n    $(SolutionDir)\\TestTools\\PortfolioOptim_SlnTest.xlsm\n\n";

    /*
    * The results are the same as those in Colab Notebook PyPortfolioOpt_FirstSteps.ipynb:
    * 
    *    import numpy as np
    *    import pandas as pd
    *
    *    # Hardcoded annual closing prices for 4 stocks over 10 years
    *    stock_data = {
    *        'Storm Fnd': [100.00,  91.94, 106.68, 118.43, 120.39, 125.24, 124.68, 140.13, 147.40, 161.05, 172.60],
    *        'Quantex  ': [100.00, 102.62, 113.92, 129.69, 130.98, 158.22, 193.35, 248.79, 278.45, 303.84, 327.84],
    *        'Polar    ': [100.00, 103.36, 120.19, 123.06, 123.51, 161.26, 144.14, 178.68, 210.01, 218.01, 289.65],
    *        'UniGlobal': [100.00, 106.67, 112.91, 121.64, 114.80, 151.17, 165.15, 222.96, 191.77, 229.97, 286.94]
    *    }
    *
    *    # Create a pandas DataFrame
    *    df = pd.DataFrame(stock_data)
    *
    *    # Calculate annual returns
    *    returns = df.pct_change().dropna()
    *    print("Annual Returns:");   print(returns);   print("\n")
    */

    // Calculate logarithm of relative returns per stock
    Eigen::MatrixXd logRelRet;
    simpleRelRet = CalcLogRetMatrix( mat_performanceTable, logRelRet );

    std::cout << "Logarithm of relative returns per stock:\n";
    std::cout << "Storm Fund II          :" << std::setw(7) << std::fixed << std::setprecision(2) << logRelRet.row(0) * 100 << "%\n";
    std::cout << "Quantex Global Value   :" << std::setw(7) << std::fixed << std::setprecision(2) << logRelRet.row(1) * 100 << "%\n";
    std::cout << "Polar Capital Insurance:" << std::setw(7) << std::fixed << std::setprecision(2) << logRelRet.row(2) * 100 << "%\n";
    std::cout << "UniGlobal              :" << std::setw(7) << std::fixed << std::setprecision(2) << logRelRet.row(3) * 100 << "%\n";

    return 0;

}  // end test_CalcRelRetMatrices
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int test_CalcCovarianceMatrix()
{
    // Performances in % (01.2015-01.2025) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets = 4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)
    Eigen::MatrixXd mat_performanceTable(numAssets, numQuotes);
    mat_performanceTable <<
               0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,  0.4740,  0.6105,  0.7260,
               0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784,
               0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965,
               0,  0.0667,  0.1291,  0.2164,  0.1480,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694;

    // Calculate simple relative returns per stock (for results see: test_CalcRelRetMatrices)
    Eigen::MatrixXd simpleRelRet;
    simpleRelRet = CalcSimpleRelRetMatrix( mat_performanceTable, simpleRelRet );

    MatrixXd covMat( numAssets, numAssets );
    bool sampleCov = true;  // Use sample covariance (N-1) instead of population covariance (N)
    covMat = CalcCovarianceMatrix( simpleRelRet.transpose(), covMat, sampleCov );

    std::cout << "Covariance matrix:\n";
    PrintMatrix( covMat, 9, 5 );

    std::cout << "Correlation matrix:\n";

    MatrixXd corrMat( numAssets, numAssets );
    corrMat = getCorrMatrix( covMat, corrMat );
    PrintMatrix( corrMat, 9, 5 );

    /* Results identical to those in Colab Notebook PyPortfolioOpt_FirstSteps.ipynb:
    
    import numpy as np
    import pandas as pd
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import sample_cov, cov_to_corr
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt

    # Hardcoded annual closing prices for 4 stocks over 10 years
    stock_data = {
        'Storm Fnd': [100.00,  91.94, 106.68, 118.43, 120.39, 125.24, 124.68, 140.13, 147.40, 161.05, 172.60],
        'Quantex  ': [100.00, 102.62, 113.92, 129.69, 130.98, 158.22, 193.35, 248.79, 278.45, 303.84, 327.84],
        'Polar    ': [100.00, 103.36, 120.19, 123.06, 123.51, 161.26, 144.14, 178.68, 210.01, 218.01, 289.65],
        'UniGlobal': [100.00, 106.67, 112.91, 121.64, 114.80, 151.17, 165.15, 222.96, 191.77, 229.97, 286.94]
    }

    # Create a pandas DataFrame
    df = pd.DataFrame(stock_data)

    # Calculate annual returns
    returns = df.pct_change().dropna()
    print("Annual Returns:");   print(returns);   print("\n")

    print("Covariance Matrix with sample_cov(df):");   print(S);   print("\n")
    S = sample_cov(returns, returns_data=True, frequency=1)
    print("Covariance Matrix with sample_cov(returns, returns_data=True, frequency=1):");   print(S);   print("\n")
    S = returns.cov()
    print("Covariance Matrix with returns.cov():");   print(S);   print("\n")
    */

    return 0;

}  // end test_CalcCovarianceMatrix
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <Eigen/Dense>
#include <random>
#include <iostream>

int test_GenerateMatrixWithNormalizedColumns() 
{
    int rows = 5;
    int cols = 3;
    unsigned int seedValue = 42; // Example seed value

    // Generate with a specific seed
    Eigen::MatrixXd normalizedMatrixSeeded = generateWeightVecMat(rows, cols, seedValue);
    std::cout << "Generated and Normalized Matrix (with seed " << seedValue << "):\n";
    std::cout << std::fixed << std::setprecision(5) << normalizedMatrixSeeded << std::endl;
    std::cout << "\nColumn Sums:\n";
    std::cout << normalizedMatrixSeeded.colwise().sum() << std::endl;

    std::cout << "\n--------------------\n";

    // Generate without a seed (will be different each time)
    Eigen::MatrixXd normalizedMatrixNoSeed = generateWeightVecMat(rows, cols);
    std::cout << "Generated and Normalized Matrix (without seed):\n";
    PrintMatrix( normalizedMatrixNoSeed, 9, 5 );
    std::cout << "Column Sums:\n";
    std::cout << normalizedMatrixNoSeed.colwise().sum() << std::endl;

    return 0;

} // end test_GenerateMatrixWithNormalizedColumns
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
static double RetIntervUpbTestCase( double risk )
{
    // Return (gain) interval upper bound function for the test case implemented
    // in test_populateRiskRetVec_RandomlyNotWeightRelated().

    return  -1.5*risk*risk + 6.5*risk - 4;
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int test_populateRiskRetVec_RandomlyNotWeightRelated()
{
    //-> Test Concept:
    //   If you generate a large number of random points uniformly within the area under 
    //   a curve y=f(x) over an interval [a,b], then:
    //   •	The average of the x(i) values will approximate the x-coordinate of the centroid.
    //   •	The average of the y(i) values will approximate the y-coordinate of the centroid.
    //   
    //   Analytical controid calculation for f(x) = -1.5*x^2 + 6.5*x - 4 over [1,3] yields:
    //   •	Centroid x-coordinate = 2.066666666666667
    //   •	Centroid y-coordinate = 1.306666666666662
    // 
    //   (For the analytical calculation see: PortfolioOptim_SlnNotes.docx in ..\TestTools\)

    const size_t nRiskRetTuples = 1000;
    RiskRetVector riskRetVec;
    const RiskInterval riskInterv = { 1.0, 3.0 };
    std::function<double(double)> retIntervUpbFunc = RetIntervUpbTestCase;
    riskRetVec = populateRiskRetVecTestCase( riskRetVec, riskInterv, retIntervUpbFunc, nRiskRetTuples );

    double riskAvg = 0;
    double retAvg  = 0;

    for ( const RiskReturnPair& pair : riskRetVec ) 
    {
        riskAvg += pair._risk;
        retAvg  += pair._return;
    }
    
    const size_t n = riskRetVec.size();
    riskAvg /= n;
    retAvg  /= n;

    std::cout << "\nCentroid of all risk-return tuples:\n";
    std::cout << "( " << riskAvg << ", " << retAvg << " )\n\n";

    return 0;

}  // end test_populateRiskRetVec_RandomlyNotWeightRelated
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <boost/math/special_functions/binomial.hpp>

int test_populateSimplexGrid() 
{
    size_t mAssets     =  4;   // number of weights
    size_t kResolution =  4;  // grid resolution (higher = more vectors)

    //-> Number of possible weight vectors for mAssets with kResolution is given by 
    //   the binomial coefficient (kResolution+mAssets-1 over mAssets-1).
    double nWeightSets = boost::math::binomial_coefficient<double>( kResolution+mAssets-1, mAssets-1 );
    std::cout << "Number of possible weight vectors for " << mAssets
              << " assets with resolution " << kResolution << " is: " << nWeightSets << "\n\n";

    auto weight_vectors = generate_simplex_grid( mAssets, kResolution );

    std::cout << "Generated " << weight_vectors.size() << " weight vectors:\n";

    for ( const auto& vec : weight_vectors )  
        std::cout<< std::fixed << std::setprecision(4) << std::setw(8) << vec.transpose() << "\n";

    return 0;
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int test_getEfficientFrontier_FromRandomDataNotWeightRelated()
{
    //-> Calculate the efficient frontier from a set of random risk-return tuples,
    //   which are not related to a portfolio with given assets and randomly generated
    //   weight sets.

    //-> Generate random risk-return pairs above risk interval [1,3] and return interval
    //   [-1.5*risk^2 + 6.5*risk - 4, 0] (see RetIntervUpbTestCase).
    const size_t nRiskRetTuples = 1000;
    RiskRetVector riskRetVec;
    const RiskInterval riskInterv = { 1.0, 3.0 };
    std::function<double(double)> retIntervUpbFunc = RetIntervUpbTestCase;
    riskRetVec = populateRiskRetVecTestCase( riskRetVec, riskInterv, retIntervUpbFunc, nRiskRetTuples, 47110815 );

    //-> From the previously generated random risk-return tuples, get the one with the
    //   maximum return for each risk subinterval, resulting from a subdivision of
    //   interval [1,3] into nRiskBars.
    const size_t nRiskBars = 20; // Number of risk subintervals 
    RiskRetVector riskRetTuplesBelowFrontier;
    riskRetTuplesBelowFrontier = 
        GetRiskReturnTuplesBelowFrontier( riskRetVec, riskRetTuplesBelowFrontier, riskInterv, nRiskBars );
   
    for ( const RiskReturnPair& pair : riskRetTuplesBelowFrontier ) 
    {
        std::cout << std::fixed << std::setprecision(6) << std::setw(9) 
                  << pair._risk << ", " << pair._return << std::endl;
    }

    return 0;

}  // end test_getEfficientFrontier_FromRandomDataNotWeightRelated
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
int test_getRiskReturnTupleForGivenWeightSet()
{   
    const size_t numAssets = 4;

    //-> Average performance per stock
    Eigen::VectorXd avgPerfVec(numAssets);
        avgPerfVec << 0.050469741, 0.1235332, 0.09260982, 0.096254528;

    //-> Covariance matrix of the stock performances
    Eigen::MatrixXd mat_portfolioCov(numAssets, numAssets);
    mat_portfolioCov << 0.0049,	0.0021,	0.0038,	0.0029,
                        0.0021,	0.0077,	0.0028,	0.0077,
                        0.0038,	0.0028,	0.0206,	0.0114,
                        0.0029,	0.0077,	0.0114,	0.0247;

    /*
    //-> Generate weight matrix (columns normalized) with a specific seed 
    const size_t  nMCSamples = 1000;
    const size_t& nWeights   = numAssets;
    Eigen::MatrixXd weightMat(nWeights, nMCSamples);
    weightMat = getWeights( weightMat );
    */
    //-> Generate weight matrix (columns normalized) with a specific seed 
    const size_t  nMCSamples = 1000;
    const size_t& nWeights   = numAssets;
    unsigned int seedValue = 47110815; // Seed value guarantees reproducibility of the results
    Eigen::MatrixXd weightMat = generateWeightVecMat( nWeights, nMCSamples, seedValue );


    //-> Populate the risk-return vector with random risk-return results by applying the
    //   random weight sets to the individual average stock performances and to the covariance matrix
    RiskRetVector riskRetVec(nMCSamples);

    for ( int j = 0; j < nMCSamples; j++ ) 
    {
        auto weights = weightMat.col(j); 

        std::cout << std::fixed << std::setprecision(6) << std::setw(9) << weights.transpose();

        //-> Calculate the average performance of the portfolio 
        riskRetVec[j]._return = CalcAvgPortPerf( avgPerfVec, weights );

        //-> Calculate the average risk of the portfolio
        riskRetVec[j]._risk = CalcAvgPortVola( mat_portfolioCov, weights );

        std::cout << std::fixed << std::setprecision(6) << std::setw(9) << riskRetVec[j]._return 
                  << std::fixed << std::setprecision(6) << std::setw(9) << riskRetVec[j]._risk << std::endl;
    }

    std::cout << "Risk-Return Pairs:\n";

    std::cout << std::fixed << std::setprecision(6) << std::setw(9)
              << "Risk" << ", " << "Return" << std::endl;

    for ( const RiskReturnPair& pair : riskRetVec )
    {
        std::cout << std::fixed << std::setprecision(6) << std::setw(9)
                  << pair._risk << ", " << pair._return << std::endl;
    }

    return 0;

}  // end test_getRiskReturnTupleForGivenWeightSet
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
int test_getEfficientFrontier_FromRandomWeights()
{
    //->Test example:
    //  Performances in % (01.2015-01.2025) for Storm Fund II, Quantex Global Value, Polar Capital Insurance, UniGlobal
    const size_t numAssets = 4;
    const size_t numQuotes = 11;    // including starting point (e.g. =11 for 10 EOY values)
    Eigen::MatrixXd mat_performanceTable(numAssets, numQuotes);
    mat_performanceTable <<
               0, -0.0806,  0.0668,  0.1843,  0.2039,  0.2524,  0.2468,  0.4013,  0.4740,  0.6105,  0.7260,
               0,  0.0262,  0.1392,  0.2969,  0.3098,  0.5822,  0.9335,  1.4879,  1.7845,  2.0384,  2.2784,
               0,  0.0336,  0.2019,  0.2306,  0.2351,  0.6126,  0.4414,  0.7868,  1.1001,  1.1801,  1.8965,
               0,  0.0667,  0.1291,  0.2164,  0.1480,  0.5117,  0.6515,  1.2296,  0.9177,  1.2997,  1.8694;
    
    //-> Calculate average performance per stock
    Eigen::VectorXd avgPerfVec(numAssets);
    avgPerfVec = CalcAvgStockPerformances( mat_performanceTable, avgPerfVec );

    // Calculate simple relative returns per stock (for results see: test_CalcRelRetMatrices)
    Eigen::MatrixXd simpleRelRet;
    simpleRelRet = CalcSimpleRelRetMatrix( mat_performanceTable, simpleRelRet );

    //-> Calculate the covariance matrix of the stock performances
    Eigen::MatrixXd mat_portfolioCov(numAssets, numAssets);
    mat_portfolioCov = CalcCovarianceMatrix( simpleRelRet.transpose(), mat_portfolioCov );

    std::cout << "Covariance matrix:\n";
    PrintMatrix( mat_portfolioCov, 9, 5 );

    //-> Generate weight matrix (columns normalized) with a specific seed 
    const size_t  nMCSamples = 1000;
    const size_t& nWeights   = numAssets;
    unsigned int seedValue = 47110815; // Seed value guarantees reproducibility of the results
    Eigen::MatrixXd weightMat = generateWeightVecMat( nWeights, nMCSamples, seedValue );

    //-> Populate the risk-return vector with random risk-return results by applying the
    //   random weight sets to the individual average stock performances and to the covariance matrix
    RiskRetVector riskRetVec(nMCSamples);

    for ( int j = 0; j < nMCSamples; j++ ) 
    {
        auto weights = weightMat.col(j); 

        //-> Calculate the average performance of the portfolio 
        riskRetVec[j]._return = CalcAvgPortPerf( avgPerfVec, weights );

        //-> Calculate the average risk of the portfolio
        riskRetVec[j]._risk = CalcAvgPortVola( mat_portfolioCov, weights );
    }

    std::cout << "Risk-Return Pairs:\n";

    std::cout << std::fixed << std::setprecision(6) << std::setw(9)
              << "Risk" << ", " << "Return" << std::endl;

    for ( const RiskReturnPair& pair : riskRetVec )
    {
        std::cout << std::fixed << std::setprecision(6) << std::setw(9)
                  << pair._risk << ", " << pair._return << std::endl;
    }
    return 0;

}  // end test_getEfficientFrontier_FromRandomWeights
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
 
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int test_find_k()
{
    PortfolioOptimizer optimizer( 15, 818800000 );
    optimizer.initialize();

    return 0;

}  // end test_find_k
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<