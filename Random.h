/*
 * Copyright (c) 2004-2005 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * MIT grants permission to use, copy, modify, and distribute this software and
 * its documentation for NON-COMMERCIAL purposes and without fee, provided that
 * this copyright notice appears in all copies.
 *
 * MIT provides this software "as is," without representations or warranties of
 * any kind, either expressed or implied, including but not limited to the
 * implied warranties of merchantability, fitness for a particular purpose, and
 * noninfringement.  MIT shall not be liable for any damages arising from any
 * use of this software.
 *
 * Author: Alexandr Andoni (andoni@mit.edu), Piotr Indyk (indyk@mit.edu)
 */

#ifndef RANDOM_INCLUDED
#define RANDOM_INCLUDED

void initRandom();

int genRandomInt(int rangeStart, int rangeEnd);

Uns32T genRandomUns32(Uns32T rangeStart, Uns32T rangeEnd);

double genUniformRandom(double rangeStart, double rangeEnd);

double genGaussianRandom();

double genCauchyRandom();

// Calculates the pdf of the Guassian distribution
double normal_pdf(double _x, double _u, double _sigma);

// Calculates the cdf of the normal distribution (mean 0, var 1)
double normal_cdf(double _x, double _step = 0.001);

#endif
