/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Antonin Portelli (antonin.portelli@ed.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/*
 * Out-of-line wrappers around the particle interaction kernels of the
 * configured SWIFT build. This file is only meant to be compiled to an object
 * file whose disassembly is analysed by kernel_flops.py to count the
 * floating-point operations of one kernel evaluation.
 *
 * Kernels with a data-dependent branch are wrapped once per branch: the
 * wrapper tests the same condition as the kernel so that the optimiser
 * removes the other path. Each wrapper is named kf_<kernel>[_<path>].
 */

#include <config.h>

/* Local includes. */
#include "swift.h"

/* The gravity kernels are not part of the public swift.h */
#include "gravity_iact.h"

/* Keep the wrappers out-of-line but inline everything they call */
#define KF __attribute__((noinline, used, flatten))

/* Placeholder so that the first wrapper does not share its address with the
 * section start (some disassemblers then only print the section label) */
__attribute__((used)) void kernel_flops_placeholder(void) {}

/* Hydro --------------------------------------------------------------------*/

KF void kf_hydro_density_sym(float r2, const float *dx, float hi, float hj,
                             struct part *pi, struct part *pj, float a,
                             float H) {
  runner_iact_density(r2, dx, hi, hj, pi, pj, a, H);
}

KF void kf_hydro_density_nonsym(float r2, const float *dx, float hi, float hj,
                                struct part *pi, const struct part *pj, float a,
                                float H) {
  runner_iact_nonsym_density(r2, dx, hi, hj, pi, pj, a, H);
}

#ifdef EXTRA_HYDRO_LOOP
KF void kf_hydro_gradient_sym(float r2, const float *dx, float hi, float hj,
                              struct part *pi, struct part *pj, float a,
                              float H) {
  runner_iact_gradient(r2, dx, hi, hj, pi, pj, a, H);
}

KF void kf_hydro_gradient_nonsym(float r2, const float *dx, float hi, float hj,
                                 struct part *pi, struct part *pj, float a,
                                 float H) {
  runner_iact_nonsym_gradient(r2, dx, hi, hj, pi, pj, a, H);
}
#endif

KF void kf_hydro_force_sym(float r2, const float *dx, float hi, float hj,
                           struct part *pi, struct part *pj, float a, float H) {
  runner_iact_force(r2, dx, hi, hj, pi, pj, a, H);
}

KF void kf_hydro_force_nonsym(float r2, const float *dx, float hi, float hj,
                              struct part *pi, const struct part *pj, float a,
                              float H) {
  runner_iact_nonsym_force(r2, dx, hi, hj, pi, pj, a, H);
}

/* Gravity P2P --------------------------------------------------------------*/

KF void kf_grav_p2p_full_newton(float r2, float h2, float h_inv, float h_inv3,
                                float mass, float *f, float *pot) {
  if (r2 >= h2) runner_iact_grav_pp_full(r2, h2, h_inv, h_inv3, mass, f, pot);
}

KF void kf_grav_p2p_full_soft(float r2, float h2, float h_inv, float h_inv3,
                              float mass, float *f, float *pot) {
  if (r2 < h2) runner_iact_grav_pp_full(r2, h2, h_inv, h_inv3, mass, f, pot);
}

KF void kf_grav_p2p_truncated_newton(float r2, float h2, float h_inv,
                                     float h_inv3, float mass, float r_s_inv,
                                     float *f, float *pot) {
  if (r2 >= h2)
    runner_iact_grav_pp_truncated(r2, h2, h_inv, h_inv3, mass, r_s_inv, f, pot);
}

KF void kf_grav_p2p_truncated_soft(float r2, float h2, float h_inv,
                                   float h_inv3, float mass, float r_s_inv,
                                   float *f, float *pot) {
  if (r2 < h2)
    runner_iact_grav_pp_truncated(r2, h2, h_inv, h_inv3, mass, r_s_inv, f, pot);
}

/* Gravity M2P --------------------------------------------------------------*/

KF void kf_grav_m2p_full_newton(float rx, float ry, float rz, float r2, float h,
                                float h_inv, const struct multipole *m,
                                float *fx, float *fy, float *fz, float *pot) {
  if (r2 >= h * h)
    runner_iact_grav_pm_full(rx, ry, rz, r2, h, h_inv, m, fx, fy, fz, pot);
}

KF void kf_grav_m2p_truncated_newton(float rx, float ry, float rz, float r2,
                                     float h, float h_inv, float r_s_inv,
                                     const struct multipole *m, float *fx,
                                     float *fy, float *fz, float *pot) {
  if (r2 >= h * h)
    runner_iact_grav_pm_truncated(rx, ry, rz, r2, h, h_inv, r_s_inv, m, fx, fy,
                                  fz, pot);
}

/* Gravity M2L --------------------------------------------------------------*/

KF void kf_grav_m2l_periodic(struct grav_tensor *l_b,
                             const struct multipole *m_a, const double pos_b[3],
                             const double pos_a[3],
                             const struct gravity_props *props,
                             const double dim[3], float rs_inv) {
  gravity_M2L_nonsym(l_b, m_a, pos_b, pos_a, props, /*periodic=*/1, dim,
                     rs_inv);
}

KF void kf_grav_m2l_nonperiodic(struct grav_tensor *l_b,
                                const struct multipole *m_a,
                                const double pos_b[3], const double pos_a[3],
                                const struct gravity_props *props,
                                const double dim[3], float rs_inv) {
  gravity_M2L_nonsym(l_b, m_a, pos_b, pos_a, props, /*periodic=*/0, dim,
                     rs_inv);
}
