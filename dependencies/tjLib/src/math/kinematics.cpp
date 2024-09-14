#include <cassert>
#include <iostream>
#include <vector>

#include "EigenWrapper.h"
#include "math/vector_math.h"

std::vector<Quaternion> integrate_body_angular_velocities(vectord t, std::vector<Vector3> w_b,
                                                          Quaternion ori_0 = Quaternion::Identity()) {
    assert(t.size() == w_b.size());

    vectord dts = diff(t);

    Quaternion ori                       = ori_0;
    std::vector<Quaternion> orientations = {ori_0};
    uint N                               = dts.size();
    for (uint i = 0; i < N; i++) {
        double dt            = dts[i];
        Vector3 w_normed     = w_b[i].normalized();
        double angular_speed = w_b[i].norm();
        double theta         = dt * angular_speed;
        Quaternion rotation{Eigen::AngleAxisd{theta, w_normed}};
        ori = ori * rotation;          // TODO order?
        orientations.push_back(ori);   // TODO inverse?
    }
    return orientations;
}

std::vector<Quaternion> diffOrientations(std::vector<Quaternion> orientations) {
    std::vector<Quaternion> deltaOrientations;
    uint N = orientations.size();
    for (uint i = 0; i < N - 1; i++) {
        Quaternion Rcurr            = orientations[i];
        Quaternion Rnext            = orientations[i + 1];
        Quaternion deltaOrientation = Rcurr.inverse() * Rnext;   // TODO CHECK ME
        // deltaOrientation = Rcurr.T @Rnext
        deltaOrientations.push_back(deltaOrientation);   // TODO  copy?
    }
    return deltaOrientations;
}

std::vector<Quaternion> integrateRotations(Quaternion R0, std::vector<Quaternion> rotations) {
    Quaternion totalRotation               = R0;
    std::vector<Quaternion> totalRotations = {R0};
    for (auto rotation : rotations) {
        totalRotation = totalRotation * rotation;
        totalRotations.push_back(totalRotation);
    }
    return totalRotations;
}

// TODO: Turn me into unit tests
// int main() {
//   std::vector<Vector3> w_b{
//       M_PI / 2.0 * Vector3{-1, 0, 0},
//       M_PI / 2.0 * Vector3{0, 0, 1},
//       M_PI / 2.0 * Vector3{0, 1, 0},
//       M_PI / 2.0 * Vector3{0, 0, 0},
//   };

//   vectord t{0, 1, 2, 3};

//   auto orientations = integrate_body_angular_velocities(t, w_b);
//   for (auto ori : orientations) {
//     std::cout << ori.toRotationMatrix() << std::endl;
//     std::cout << std::endl;
//   }
// }
