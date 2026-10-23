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
#ifndef SWIFT_TASK_PERF_H
#define SWIFT_TASK_PERF_H

#include <config.h>

/**
 * @brief The interaction counters recorded for each task.
 *
 * Each counter records how many times a given kernel was evaluated while
 * running a task. Combined with the cost of a single kernel evaluation
 * (see tools/task_perf) this gives the FLOPs and bytes processed by the task.
 *
 * Be sure to update task_perf_counter_names in task.c and the analysis
 * scripts in tools/task_perf if you modify this list!
 */
enum task_perf_counters {

  /*! Symmetric particle-particle hydro kernel calls (runner_iact_*) */
  task_perf_hydro_iact_sym = 0,

  /*! Non-symmetric particle-particle hydro kernel calls
   * (runner_iact_nonsym_*) */
  task_perf_hydro_iact_nonsym,

  /*! Gravity P2P evaluations with the Newtonian kernel (incl. padding) */
  task_perf_grav_p2p_full,

  /*! Gravity P2P evaluations with the mesh-truncated kernel (incl. padding) */
  task_perf_grav_p2p_truncated,

  /*! Gravity M2P evaluations with the Newtonian kernel */
  task_perf_grav_m2p_full,

  /*! Gravity M2P evaluations with the mesh-truncated kernel */
  task_perf_grav_m2p_truncated,

  /*! Gravity M2L evaluations (one per multipole-field tensor pair) */
  task_perf_grav_m2l,

  /*! Gravity particles read by P2P/M2P kernels (gravity cache fills or
   * direct reads of the source cell) */
  task_perf_grav_parts_loaded,

  task_perf_count
};

/**
 * @brief A set of #task_perf_counters.
 */
struct task_perf {
  long long counts[task_perf_count];
};

#ifdef SWIFT_TASK_PERF

/*! Add n to the counter c of the #runner r */
#define TASK_PERF_ADD(r, c, n) ((r)->perf.counts[(c)] += (n))

#else

#define TASK_PERF_ADD(r, c, n)

#endif /* SWIFT_TASK_PERF */

#endif /* SWIFT_TASK_PERF_H */
