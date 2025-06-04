#ifndef TAICHI_RHI_METAL_SINCOSPI_H
#include <math.h>

extern "C" {
void __sincospif(float x, float *sinp, float *cosp);
void __sincospi(double x, double *sinp, double *cosp);
}
#endif  // TAICHI_RHI_METAL_SINCOSPI_H
