#pragma once

#include <cstdint>
#include <optional>

#include <boost/multiprecision/cpp_int.hpp>
using boost::multiprecision::uint128_t;

#include <Eigen/Dense>
using namespace Eigen;


struct RiskInterval{ double lwb; double upb; };
struct ReturnInterval { double lwb; double upb; };
struct RiskReturnPair { double _risk; double _return; };
typedef std::vector<RiskReturnPair> RiskRetVector;


Eigen::MatrixXcd SanitizeMatrix( const Eigen::MatrixXcd& matrix );
 
Eigen::MatrixXcd SanitizeUsingLambda( const Eigen::MatrixXcd& matrix );

Eigen::VectorXd& CalcAvgStockPerformances( const Eigen::MatrixXd& perfMat, Eigen::VectorXd& stockPerformances );

double CalcAvgPortPerf( const Eigen::VectorXd& stockPerfAvg, const Eigen::VectorXd& weights );

MatrixXd& CalcSimpleRelRetMatrix( const MatrixXd& perfMat, MatrixXd& relRetMat );

MatrixXd& CalcLogRetMatrix( const MatrixXd& perfMat, MatrixXd& logRetMat );

MatrixXd& CalcCovarianceMatrix( const MatrixXd& dataMat, MatrixXd& covMat, bool sampleCov = true );

Eigen::MatrixXd getCorrMatrix( const Eigen::MatrixXd& covMat, Eigen::MatrixXd& corrMat );

double CalcAvgPortVola(const Eigen::MatrixXd& covMat, const Eigen::VectorXd& weights);

Eigen::MatrixXd generateWeightVecMat( size_t rows, size_t cols, std::optional<unsigned int> seed = std::nullopt );

void generate_compositions( size_t k, size_t m, std::vector<size_t>& current, std::vector<Eigen::VectorXd>& results ); 

std::vector<Eigen::VectorXd> generate_simplex_grid( size_t mAssets, size_t kResolution );

RiskRetVector& populateRiskRetVecTestCase( RiskRetVector& riskRetVec, 
                                           RiskInterval riskInterv, 
                                           std::function<double(double)> retIntervUpbFunc,
                                           size_t nTuples,
                                           std::optional<unsigned int> seed = std::nullopt );

RiskRetVector& GetRiskReturnTuplesBelowFrontier( RiskRetVector& riskRetVec,
                                                 RiskRetVector& riskRetTuplesBelowFrontier,
                                                 const RiskInterval riskInterv,
                                                 const size_t nRiskBars                     );

size_t safe_uint128_to_size_t( boost::multiprecision::uint128_t n );

class PortfolioOptimizer 
{
  public:
    PortfolioOptimizer( size_t m, size_t nMin = 0 ) : myNumAssets(m), myMinTotalNumOfGridPoints(nMin) {}
    // - The minimum number of simplex grid points nMin can optionally be provided.
    //   In case it will be provided, the minimum simplex grid resolution constant kMin, for which n > nMin, will be calculated.
    //   Example: m=4, nMin=1000 => kMin=17, n=1140
    //
    // - If nMIn is not provided or the calculated grid resolution constant kMin is lower than 4, 
    //   which means that the initial weight grid space width is > 25% (=100/k),
    //   then k will be set to 4, otherwise k := kMin

    size_t getMinNumGridpoints() const { return myMinTotalNumOfGridPoints; }

    void initialize();

  private:
      size_t find_k( size_t mAssets, uint128_t nGridPoints );

      size_t myNumAssets;
      size_t myMinTotalNumOfGridPoints;
      size_t mySimplexGridResolution;

};  // end class PortfolioOptimizer 

