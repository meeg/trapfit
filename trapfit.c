#include <math.h>
#include <stdio.h>

// compile with:
// gcc -shared -fPIC -o trapfit.so trapfit.c
// from https://scipy.github.io/devdocs/tutorial/integrate.html#quad-callbacks

double dc_integrand(int n, double *x) {
    double E = x[0];
    double t = x[1];
    //double dc_eq = x[2];
    double dt = x[2];
    double density = 0;
    for (int i=n-1; i>2; i--) {
        density *= E;
        density += x[i];
    }
    double temp = 170; // K
    double kt = 8.62e-5 * temp; // eV
    double prefactor = 1.6e6; // 1.6e21 * 1e-15

    double decayrate = prefactor * pow(temp,2) * exp(-E/kt); // s^-1
    return 86400 * density * decayrate * exp((dt-t) * decayrate);
}

double single_integrand(int n, double *x) {
    double E = x[0];
    double t = x[1];
    int degree = x[2];
    //double dc_eq = x[2];
    double density = pow(E, degree);
    double temp = 170; // K
    double kt = 8.62e-5 * temp; // eV
    double prefactor = 1e-15 * 1.6e21;

    double decayrate = prefactor * pow(temp,2) * exp(-E/kt); // s^-1
    return 86400 * density * decayrate * exp(-t * decayrate);
}
