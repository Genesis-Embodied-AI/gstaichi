#include <math.h>

extern "C" {
void __sincospif(float x, float *sinp, float *cosp) {
  *sinp = sinf(M_PI * x);
  *cosp = cosf(M_PI * x);
}
void __sincospi(double x, double *sinp, double *cosp) {
  *sinp = sin(M_PI * x);
  *cosp = cos(M_PI * x);
}
}
