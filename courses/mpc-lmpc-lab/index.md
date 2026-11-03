---
eyebrow: Three parts · fifteen concepts, six labs, four extensions · NumPy, SciPy, your browser
description: >-
  A course on constrained optimal control in three parts: fifteen short concept chapters, one idea each on the same
  double integrator, from state-space models and convex optimisation through invariant sets and the MPC stability
  theorem to Learning MPC and dynamic programming; six labs that assemble them, collision avoidance, Learning
  MPC on a repeated task, minimum-time racing, learned vehicle models, the bridge to reinforcement learning and a
  capstone that checks every guarantee numerically; and four chapters beyond the nominal case, tube MPC, offset-free
  tracking, nonlinear MPC by successive linearisation and a Learning MPC racing controller driving laps. Every demo
  runs in the browser and every number is reproduced by a script.
lede: >-
  Model predictive control solves an optimisation problem at every sample. This course does not treat that
  optimiser as a black box: for every controller it asks what the decision variables, objective and constraints
  are, what makes the problem feasible at the next step, why the closed loop is stable, and which assumptions a
  real system would break. Part A teaches the concepts one at a time, each on the same small system, with the
  function that computes it read line by line. Part B assembles them into complete designs, from a collision
  decision to <strong>Learning MPC</strong>, which learns its terminal ingredients from its own successful
  executions, and to the connection with dynamic programming and reinforcement learning. Part C goes beyond
  the nominal case: disturbances, setpoints, a nonlinear plant, and the racing controller that finally drives laps.
assumes: Assumes linear algebra, basic control (state-space models, stability) and Python
---

**How to read a chapter.** Every concept chapter has the same shape: a live demo under the title, the idea in plain words and then precisely, the function of the course package that computes it read line by line, the pitfalls, questions with folded answers, exercises whose solutions run, and the chapter on one page. Every number quoted on a page is printed by that chapter's script, `code/<chapter>_numbers.py`, and the browser demos use the same algorithms as the package.

**How to study each controller.** Always find four layers. The model: what are the state $x$, the input $u$ and the dynamics $f$? The optimisation: what are the decision variables, the objective and the constraints? The guarantees: what gives feasibility, safety or stability? The limitations: which assumptions would fail on a real platform? The labs are built around these four questions, and they are a good structure for reviewing any controller design.
