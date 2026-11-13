---
title: Notation and the course's workshop
---

Every chapter uses the symbols, names and numbers on this page, so that a formula in one chapter means the same
thing in all the others. Machines, jobs, families and operations are numbered **from 0**, as in the code.

## Time and units {#units}

All times are **integer minutes**, measured from the start of the simulated day, $t = 0$. A simulated day has
60 jobs and lasts about 1,350 minutes (the last job arrives near minute 1,250).

## Jobs, operations, machines {#objects}

| Symbol | Meaning | In the code |
| --- | --- | --- |
| $j \in \mathcal J = \{0, \dots, n-1\}$ | a job (an order), $n = 60$ per simulated day | `Job.idx` |
| $f(j) \in \{0, 1, 2, 3\}$ | the product family of job $j$ | `Job.family` |
| $r_j$ | release time: when job $j$ arrives at the load station | `Job.arrival` |
| $d_j$ | due date | `Job.due` |
| $(j, k)$, $k = 0, \dots, K_j - 1$ | the $k$-th operation (step) of job $j$; here $K_j = 4$ for every job | `(job, op)` |
| $m \in \mathcal M = \{0, \dots, 4\}$ | a machine | machine index |
| $\mathcal M_{jk} \subseteq \mathcal M$ | the machines able to process operation $(j,k)$: 1 to 3 of them | `ws.alternatives(f, k)` |
| $p_{jkm}$ | processing time of $(j,k)$ on machine $m$ | `ws.proc(f, k, m)` |
| $s_m(f, g)$ | changeover time on machine $m$ from family $f$ to family $g$; $s_m(f,f)=0$; a machine that has run nothing needs none | `ws.setup_time(m, f, g)` |
| $\tau(a, b)$ | travel time of a cart between stations $a$ and $b$ (load station 0, machine $m$ at station $m+1$) | `ws.travel[a][b]` |
| $\delta = 2$ | the allowance the CP-SAT planner adds to each move for a cart to come | `allowance` |

## Schedules and their measures {#measures}

| Symbol | Meaning |
| --- | --- |
| $S_{jk}$, $E_{jk}$ | start and end of operation $(j,k)$ (the changeover, if any, comes before $S_{jk}$) |
| $C_j = E_{j, K_j-1}$ | completion time of job $j$ |
| $L_j = C_j - d_j$ | lateness (negative when early) |
| $T_j = \max(0, L_j)$ | tardiness: how late, or 0 if on time |
| $F_j = C_j - r_j$ | flow time: time spent in the shop |
| $C_{\max} = \max_j C_j$ | makespan |
| $\bar T = \frac1n \sum_j T_j$ | **mean tardiness**, the headline measure of the course ("average lateness per order" in plain words) |

The CP-SAT model minimises $10 \sum_j T_j + \sum_j C_j$ over the jobs it plans: tardiness first, and among equally
late plans, earlier completions.

## Randomness {#random}

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\lambda$ | arrival rate; gaps between arrivals are exponential with mean $1/\lambda$ | $1/\lambda = 20.83$ min |
| $\rho$ | load: fastest-route work arriving per minute, divided by the number of machines | $0.60$ |
| MTBF | mean time between a repair and the next failure, per machine ("frequent breakdowns") | 120 min |
| MTTR | mean repair time; a repair lasts uniform $[0.5, 1.5] \times$ MTTR | 20 min |
| $A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$ | availability of a machine | $120/140 = 0.857$, about 14 % downtime |
| seed $i = 0, \dots, 19$ | the 20 test days; seeds 5000–5009 validate, seeds from 100000 train the agent | |

## Decision methods and their names {#methods}

Every chapter uses these names. The plain name comes first; the abbreviation follows in brackets the first time.

| Plain name | Abbreviation | What it does |
| --- | --- | --- |
| First come, first served | FIFO | the job that arrived first |
| Shortest job first | SPT | the shortest processing time |
| Most urgent first | EDD | the earliest due date |
| Shortest setup first | SST | the job needing the shortest changeover ("avoid colour changes" in the paint example) |
| Self-learning agent | DQN | a Double DQN that picks one of the four rules at each decision |
| Quick re-planning | Greedy replan | the re-planning loop with a greedy plan, no solver (an ablation) |
| Optimiser re-planning | CP-SAT | the re-planning loop with the CP-SAT model at every event |

Routing (which machine a job goes to) is the same for the four rules and the agent: the eligible machine with the
earliest expected finish.

## Reinforcement learning {#rl}

| Symbol | Meaning |
| --- | --- |
| $s_t$, $a_t$, $r_{t+1}$ | state, action and reward at decision $t$ |
| $a_t \in \{\text{FIFO}, \text{SPT}, \text{EDD}, \text{SST}\}$ | the action: which rule picks the next job |
| $\gamma = 0.97$ | discount factor |
| $Q(s, a)$, $Q_\theta$ | action value; the network with weights $\theta$ that approximates it |

## Statistics {#stats}

| Symbol | Meaning |
| --- | --- |
| $\Delta_i$ | paired difference on test day $i$, e.g. $\bar T^{\text{CP-SAT}}_i - \bar T^{\text{SST}}_i$ |
| $B = 10{,}000$ | bootstrap resamples; intervals are 95 % percentile intervals |

## The worked example: one paint machine {#paint}

Used in chapters A1 and A2, and on the landing page. One machine paints parts; it is set up for **blue**; any
colour change takes **10 minutes**. Four orders wait, in this order of arrival:

| Order | Colour | Painting time | Due at minute |
| --- | --- | --- | --- |
| A | blue | 20 | 30 |
| B | red | 10 | 25 |
| C | blue | 15 | 70 |
| D | red | 25 | 60 |

Total tardiness: FIFO (ABCD) 55, SPT (BCAD) 75, EDD (BADC) 85, SST (ACBD) 50, best of all 24 orders (ABDC) 40
minutes. `JS.PAINT` in `interactive/jobshop-lib.js` holds it; `scripts/jobshop_toy_example.py` computes it.

## The course's workshop {#workshop}

The repository's `make_workshop()` with its default seed. Every chapter that needs real data uses these numbers
(`JS.WORKSHOP` in the shared toolkit holds them).

Routings: for each family, the four operations and, for each, the eligible machines with their processing times
in minutes.

| Family | Operation 0 | Operation 1 | Operation 2 | Operation 3 | Fastest route |
| --- | --- | --- | --- | --- | --- |
| 0 | m1: 6, m2: 10 | m2: 26, m3: 30 | m1: 10, m4: 7 | m0: 8, m2: 8, m3: 7 | 46 |
| 1 | m0: 9 | m1: 18, m3: 18 | m2: 22, m4: 19 | m2: 15, m3: 17 | 61 |
| 2 | m0: 16, m2: 19, m3: 16 | m1: 17, m3: 24, m4: 24 | m1: 18, m4: 21 | m0: 21, m4: 19 | 70 |
| 3 | m1: 26, m4: 23 | m4: 17 | m0: 20, m1: 16, m2: 20 | m0: 17, m2: 19 | 73 |

Changeovers $s_m(f, g)$ range from 3 to 10 minutes and are asymmetric; travel times between stations are 2 or 3
minutes. Due dates: $d_j = r_j + k_j \cdot W_{f(j)}$ with $W_f$ the fastest-route work above and $k_j$ uniform in
$[2, 4]$, rounded to the minute.

## The headline results {#results}

From `results/runs.csv` of the repository (20 test days, 60 jobs a day). Mean tardiness in minutes, with 95 %
bootstrap intervals:

| Method | No breakdowns | Frequent breakdowns |
| --- | --- | --- |
| FIFO | 15.5 [7.6, 26.7] | 60.8 [41.9, 82.2] |
| SPT | 10.0 [5.4, 16.3] | 45.4 [32.0, 60.8] |
| EDD | 7.0 [3.3, 12.4] | 37.6 [27.1, 50.0] |
| SST | 9.4 [5.2, 15.2] | 34.1 [24.8, 44.9] |
| DQN | 7.4 [4.0, 12.1] | 35.4 [25.1, 47.0] |
| Greedy replan | 3.1 [1.3, 5.3] | 24.0 [15.8, 33.6] |
| CP-SAT | 1.3 [0.2, 2.9] | 19.3 [12.3, 27.1] |

Paired differences, CP-SAT minus the best rule: $-5.7\ [-9.6, -2.9]$ and $-14.8\ [-18.4, -11.7]$ minutes, better on
20 of 20 days at both levels; CP-SAT reduces mean tardiness by 43 % under breakdowns.
