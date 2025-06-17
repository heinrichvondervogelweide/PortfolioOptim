#pragma once

#pragma warning( disable : 4267 )

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


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <stdexcept>

template<typename T, T Min, T Max> class Range 
{
  public:
    Range( T val ) { if (val < Min || val > Max)  throw std::out_of_range("Value out of range"); value = val; }
    operator T() const { return value; }  // implicit conversion

  private:
    T value;

};  // end class Range
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// #include <Eigen/Dense>
#include <string>
#include <stdexcept>

class SharpeRankTable 
{
    //-> This class is an implementation of the table on sheet "OptimOnSmplxGrid 4 Assets"
    //   of workbook PortfolioOptim_SlnTest.xlsm, which has got three column sections with 
    //   weights. The leftmost weight secton drives the calculation of portfolio return,
    //   risk, and Sharpe Ratio, which are stored in this order right of the driving weight
    //   section. On the right side of the Sharpe Ratio column follows another section of
    //   downscaled weights (meaning a shrinkage of the simplex grid). The third rightmost
    //   weight section contains the downscaled weights after normalization.

  public:
 
    SharpeRankTable( size_t nSmplxGridPoints, size_t nAssets, std::string tableName = "" )
        : myNumWeightGridPoints(nSmplxGridPoints),
          myNumAssets(nAssets),
          myName(std::move(tableName)),
          myTable(nSmplxGridPoints, 3*nAssets + 3)
    {
        if ( nSmplxGridPoints < 4 ) throw std::invalid_argument("nSmplxGridPoints must have a minimum size (TBD)");
    }

    //-> Block access to entire weight sections. 
    //   Eigen::Block<Eigen::MatrixXd> creates a reference to a subregion of a matrix. 
    //   Modifying the returned subregion will modify the corresponding region in the original matrix.
    Eigen::Block<Eigen::MatrixXd> block_DrivingWeights   () { return myTable.block(0,        0         , myTable.rows(), myNumAssets); }
    Eigen::Block<Eigen::MatrixXd> block_DownscaledWeights() { return myTable.block(0,   myNumAssets + 3, myTable.rows(), myNumAssets); }
    Eigen::Block<Eigen::MatrixXd> block_NormalizedWeights() { return myTable.block(0, 2*myNumAssets + 3, myTable.rows(), myNumAssets); }

    //-> Block access to Return, Risk, and Sharpe Ratio columns
    Eigen::Block<Eigen::MatrixXd> col_Return() { return myTable.block(0, myNumAssets,   myTable.rows(), 1); }
    Eigen::Block<Eigen::MatrixXd> col_Risk  () { return myTable.block(0, myNumAssets+1, myTable.rows(), 1); }
    Eigen::Block<Eigen::MatrixXd> col_Sharpe() { return myTable.block(0, myNumAssets+2, myTable.rows(), 1); }

    size_t getSharpeRatioColIdx() const { return myNumAssets + 2; }  // Index of the Sharpe Ratio column in the table

    const Eigen::MatrixXd& getTable() const { return myTable; }

    void sort();  // Sort the table by Sharpe Ratio in descending order

    Eigen::Index rows() const { return myTable.rows(); }
    Eigen::Index cols() const { return myTable.cols(); }

  private:
    Eigen::MatrixXd myTable;                // Matrix to hold the table data
    std::string     myName;                 // Optional name for the table
    size_t          myNumAssets;            // Size of weight vectors
    size_t          myNumWeightGridPoints;  // Number of weight grid points

};  // end class SharpeRankTable
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#include <boost/multiprecision/cpp_int.hpp>

class PortfolioOptimizer 
{
  public:
    PortfolioOptimizer( size_t m, size_t nMin = 0 ) 
        : myNumAssets(m), 
          myMinWeight(m),
          myMaxWeight(m),
          myMinTotalNumOfGridPoints(nMin)
    {}
    // - The minimum number of simplex grid points nMin can optionally be provided.
    //   In case it will be provided, the minimum simplex grid resolution constant kMin, for which n > nMin, will be calculated.
    //   Example: m=4, nMin=1000 => kMin=17, n=1140
    //
    // - If nMIn is not provided or the calculated grid resolution constant kMin is lower than 4, 
    //   which means that the initial weight grid space width is > 25% (=100/k),
    //   then k will be set to 4, otherwise k will be set to the calculated kMin.

    ~PortfolioOptimizer() { if (p_myRankTable) delete p_myRankTable; p_myRankTable = nullptr; }
    
    size_t getMinNumGridpoints() const { return myMinTotalNumOfGridPoints; }

    const SharpeRankTable* getRankTable() const { return p_myRankTable; }

    void initialize( Eigen::VectorXd avgPerfVec, Eigen::MatrixXd portfolioCov );

    void iterate();

    size_t find_k( size_t mAssets, uint128_t nGridPoints );

  private:
    PortfolioOptimizer(); // Prevent usage of default constructor 

    void CalcPerformance();

    void getMinMaxWeights();

    void downscaleWeightRanges();

    void normalizeWeights();

    double topWeightSpread();

    size_t myNumAssets;
    Eigen::VectorXd myAvgPerfVec;
    Eigen::MatrixXd myPortfolioCov;

    size_t myMinTotalNumOfGridPoints;
    size_t mySimplexGridResolution;

    uint128_t myNumWeightVectors;

    SharpeRankTable* p_myRankTable;  // Table to hold the Return, Risk, Sharpe Ratio results, and weight vectors

    std::vector<double> myMinWeight;
    std::vector<double> myMaxWeight;

};  // end class PortfolioOptimizer 
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

