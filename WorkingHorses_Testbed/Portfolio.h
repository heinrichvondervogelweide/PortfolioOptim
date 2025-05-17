#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <optional>

#include <Eigen/Dense>

Eigen::MatrixXd generateAndNormalize( size_t rows, size_t cols, std::optional<unsigned int> seed = std::nullopt );


class Portfolio
{
  public:
    Portfolio() {}
    ~Portfolio() {};

  private:

};  // end class Portfolio
