#pragma once

#include <Eigen/Dense>


void PrintMatrix( Eigen::MatrixXd mat, size_t width = 8, size_t precision = 2 );

int test_SanitizeMatrix();

int test_GenerateMatrixWithNormalizedColumns();

int test_populateSimplexGrid();

int test_CalculateAvgPortfolioPerf();

int test_CalcRelRetMatrices();

int test_CalcCovarianceMatrix();

int test_getRiskReturnTupleForGivenWeightSet();

int test_getEfficientFrontier_FromRandomWeights();

int test_populateRiskRetVec_RandomlyNotWeightRelated();

int test_getEfficientFrontier_FromRandomDataNotWeightRelated();

int test_find_k();
