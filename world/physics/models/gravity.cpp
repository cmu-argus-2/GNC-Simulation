#include "gravity.h"

#include <iostream>
#include "math/EigenWrapper.h"
#include <math.h>


Vector3 gravitational_acceleration(const Vector3 r) 
{
    return spherical_acceleration(r) + J2_perturbation(r);
}

Vector3 spherical_acceleration(const Vector3 r)
{
    Vector3 acceleration = - (mu/pow(r.norm(), 3.0))*r;

    return acceleration;
}

Vector3 J2_perturbation(const Vector3 r)
{
    Vector3 acceleration;

    double J2_factor = 3*J2*mu*pow(R_earth, 2)/(2*pow(r.norm(), 5));

    acceleration(0) = (5*pow(r(2),2)/pow(r.norm(),2) - 1)*r(0)*J2_factor;
    acceleration(1) = (5*pow(r(2),2)/pow(r.norm(),2) - 1)*r(1)*J2_factor;
    acceleration(2) = (5*pow(r(2), 2)/pow(r.norm(), 2) - 3)*r(2)*J2_factor;

    return acceleration;
}