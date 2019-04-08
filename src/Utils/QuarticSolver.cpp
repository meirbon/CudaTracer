/*------------------------------------------------------------------------------
 * quarticsolver.cpp
 *
 * Class describing a quartic equation
 * Copyright Jonathan Zrake, NYU CCPP, https://github.com/jzrake/Mara
 *------------------------------------------------------------------------------
 */

#include "QuarticSolver.h"
#include <cmath>

int QuarticEquation::Solve(double *const roots) const
{
    double r1, r2, r3, r4;
    int nr12, nr34;

    const double a3 = d3 / d4;
    const double a2 = d2 / d4;
    const double a1 = d1 / d4;
    const double a0 = d0 / d4;

    const double au2 = -a2;
    const double au1 = (a1 * a3 - 4.0 * a0);
    const double au0 = 4.0 * a0 * a2 - a1 * a1 - a0 * a3 * a3;

    double x1, x2, x3;
    int nr;
    solve_cubic_equation(1.0, au2, au1, au0, x1, x2, x3, nr);

    double u1;
    if (nr == 1)
    {
        u1 = x1;
    }
    else
    {
        u1 = (x1 > x3) ? x1 : x3;
    }

    const double R2 = 0.25 * a3 * a3 + u1 - a2;
    const double R = (R2 > 0.0) ? sqrt(R2) : 0.0;

    double D2, E2;
    if (R != 0.0)
    {
        const double foo1 = 0.75 * a3 * a3 - R2 - 2.0 * a2;
        const double foo2 =
            0.25 * (4.0 * a3 * a2 - 8.0 * a1 - a3 * a3 * a3) / R;
        D2 = foo1 + foo2;
        E2 = foo1 - foo2;
    }
    else
    {
        const double foo1 = 0.75 * a3 * a3 - 2.0 * a2;
        const double foo2 = 2.0 * sqrt(u1 * u1 - 4.0 * a0);
        D2 = foo1 + foo2;
        E2 = foo1 - foo2;
    }

    if (D2 >= 0.0)
    {
        const double D = sqrt(D2);
        r1 = -0.25 * a3 + 0.5 * R - 0.5 * D;
        r2 = -0.25 * a3 + 0.5 * R + 0.5 * D;
        nr12 = 2;
    }
    else
    {
        r1 = r2 = -0.25 * a3 + 0.5 * R;
        nr12 = 0;
    }

    if (E2 >= 0.0)
    {
        const double E = sqrt(E2);
        r3 = -0.25 * a3 - 0.5 * R - 0.5 * E;
        r4 = -0.25 * a3 - 0.5 * R + 0.5 * E;
        nr34 = 2;
    }
    else
    {
        r3 = r4 = -0.25 * a3 - 0.5 * R;
        nr34 = 0;
    }

    int i = 0;
    if (nr12 != 0)
    {
        roots[i++] = r1;
        roots[i++] = r2;
    }
    if (nr34 != 0)
    {
        roots[i++] = r3;
        roots[i++] = r4;
    }

    return nr12 + nr34;
}

void QuarticEquation::solve_cubic_equation(double c3, double c2, double c1,
                                           double c0, double &x1, double &x2,
                                           double &x3, int &nr) const
{
    const double a2 = c2 / c3;
    const double a1 = c1 / c3;
    const double a0 = c0 / c3;

    const double q = a1 / 3.0 - a2 * a2 / 9.0;
    const double r = (a1 * a2 - 3.0 * a0) / 6.0 - a2 * a2 * a2 / 27.0;
    const double delta = q * q * q + r * r;

    if (delta > 0.0)
    {
        double s1 = r + sqrt(delta);
        s1 = (s1 >= 0.0) ? pow(s1, 1. / 3.) : -pow(-s1, 1. / 3.);

        double s2 = r - sqrt(delta);
        s2 = (s2 >= 0.0) ? pow(s2, 1. / 3.) : -pow(-s2, 1. / 3.);

        x1 = (s1 + s2) - a2 / 3.0;
        x2 = x3 = -0.5 * (s1 + s2) - a2 / 3.0;

        nr = 1;
    }
    else if (delta < 0.0)
    {
        const double theta = acos(r / sqrt(-q * q * q)) / 3.0;
        const double costh = cos(theta);
        const double sinth = sin(theta);
        const double sq = sqrt(-q);

        x1 = 2.0 * sq * costh - a2 / 3.0;
        x2 = -sq * costh - a2 / 3.0 - sqrt(3.) * sq * sinth;
        x3 = -sq * costh - a2 / 3.0 + sqrt(3.) * sq * sinth;

        nr = 3;
    }
    else
    {
        const double s = (r >= 0.0) ? pow(r, 1. / 3.) : -pow(-r, 1. / 3.);
        x1 = 2.0 * s - a2 / 3.0;
        x2 = x3 = -s - a2 / 3.0;
        nr = 3;
    }
}