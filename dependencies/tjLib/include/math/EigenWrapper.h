#ifndef _EKF_EIGEN_WRAPPER_
#define _EKF_EIGEN_WRAPPER_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// convenient typedefs for commonly used Eigen types
typedef Eigen::Vector2d Vector2;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Matrix<double, 4, 1> Vector4;
typedef Eigen::Matrix<double, 5, 1> Vector5;
typedef Eigen::Matrix<double, 6, 1> Vector6;
typedef Eigen::Matrix<double, 8, 1> Vector8;
typedef Eigen::Matrix<double, 9, 1> Vector9;
typedef Eigen::Matrix<double, 13, 1> Vector13;
typedef Eigen::VectorXd VectorXd;

typedef Eigen::Matrix2d Matrix_2x2;
typedef Eigen::Matrix3d Matrix_3x3;
typedef Eigen::Matrix<double, 4, 4> Matrix_4x4;
typedef Eigen::Matrix<double, 5, 5> Matrix_5x5;
typedef Eigen::Matrix<double, 6, 6> Matrix_6x6;
typedef Eigen::Matrix<double, 8, 8> Matrix_8x8;
typedef Eigen::Matrix<double, 9, 9> Matrix_9x9;
typedef Eigen::Matrix<double, 18, 18> Matrix_18x18;

typedef Eigen::Matrix<double, 2, 3> Matrix_2x3;
typedef Eigen::Matrix<double, 2, 9> Matrix_2x9;
typedef Eigen::Matrix<double, 3, 2> Matrix_3x2;
typedef Eigen::Matrix<double, 3, 4> Matrix_3x4;
typedef Eigen::Matrix<double, 3, 6> Matrix_3x6;
typedef Eigen::Matrix<double, 3, 9> Matrix_3x9;
typedef Eigen::Matrix<double, 4, 3> Matrix_4x3;
typedef Eigen::Matrix<double, 6, 3> Matrix_6x3;
typedef Eigen::Matrix<double, 6, 9> Matrix_6x9;
typedef Eigen::Matrix<double, 8, 9> Matrix_8x9;
typedef Eigen::Matrix<double, 9, 2> Matrix_9x2;
typedef Eigen::Matrix<double, 9, 3> Matrix_9x3;
typedef Eigen::Matrix<double, 9, 6> Matrix_9x6;
typedef Eigen::Matrix<double, 9, 8> Matrix_9x8;
typedef Eigen::MatrixXd MatrixXd;

typedef Eigen::Quaterniond Quaternion;

typedef Eigen::Matrix<double, 13, 1> StateVector;

// Unit axes used in initialization
static const Vector3 unitX{1, 0, 0};
static const Vector3 unitY{0, 1, 0};
static const Vector3 unitZ{0, 0, 1};

Matrix_2x2 pinv_2x2(const Matrix_2x2& A, double threshold = 1e-4);
Matrix_3x3 pinv_3x3(const Matrix_3x3& A, double threshold = 1e-4);
Matrix_5x5 pinv_5x5(const Matrix_5x5& A, double threshold = 1e-4);
Matrix_6x6 pinv_6x6(const Matrix_6x6& A, double threshold = 1e-4);
Matrix_18x18 exp(const Matrix_18x18& A);
Matrix_9x9 exp_9x9(const Matrix_9x9& A);
#endif
