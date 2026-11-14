---
title: Notation and the course's workshop
---

Every chapter uses the symbols, names and numbers on this page, so that a formula in one chapter means the same
thing in all the others. Machines, jobs, families and operations are numbered **from 0**, as in the code.

## Time and units {#units}

All times are **integer minutes**, measured from the start of the simulated day, $t = 0$. A simulated day has
60 jobs and lasts about 1,350 minutes without breakdowns: the last job arrives at minute 1,208 on average over the
20 test days (between 895 and 1,491), and the shop then clears what remains.

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

## The planner's model and the snapshot {#model}

Introduced in chapter C1 (the model) and chapter C3 (the snapshot it is solved on).

| Symbol | Meaning | In the code |
| --- | --- | --- |
| $t_0$ | time of the snapshot; every planned start and end lies in $[t_0, H]$ | `snap.now` |
| $H$ | the planning horizon | `horizon` |
| $\phi_m$ | when machine $m$ is expected to be free (a broken machine: the *mean* repair time) | `machine_free` |
| $g_m$ | the family machine $m$ is set up for (none if it has run nothing) | `machine_family` |
| $R_{jm}$ | ready time: the earliest minute job $j$ can be at machine $m$ for its next operation (the ready map) | `job.ready` |
| $x_{jkm}$ | 1 if operation $(j,k)$ runs on machine $m$, else 0 (presence literal of an optional interval) | per-machine literal |
| $\alpha_{jk}$ | when job $j$ reaches the machine of operation $(j,k)$; $S_{jk} \ge \alpha_{jk}$ | `a` |
| $z^m_{ab}$ | 1 if, on machine $m$, operation $b$ directly follows $a$ (an arc of the circuit) | arc literal |
| $\ell(m', m)$ | transport lag: $0$ if $m' = m$, else $\delta + \tau(\text{station of } m', \text{station of } m)$ | `transport_lag` |
| $y_{ab}$, $M$ | big-M MILP only: $y_{ab} = 1$ if $a$ comes before $b$ (not necessarily directly); $M$ the large constant | |

## Randomness {#random}

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\lambda$ | arrival rate; gaps between arrivals are exponential with mean $1/\lambda$ | $1/\lambda = 20.83$ min |
| $\rho$ | load: fastest-route work arriving per minute, divided by the number of machines | $0.60$ |
| MTBF | mean time between a repair and the next failure, per machine ("frequent breakdowns") | 120 min |
| MTTR | mean repair time; a repair lasts uniform $[0.5, 1.5] \times$ MTTR | 20 min |
| $A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$ | availability of a machine | $120/140 = 0.857$, about 14 % downtime |
| seed $i = 0, \dots, 19$ | the 20 test days; seeds 5000–5009 validate; the agent of training seed $k = 0, \dots, 4$ trains on days $100000 + 1000k + e$, episodes $e = 1, \dots, 1500$: seeds 100001 to 105500, 5,500 distinct days, since consecutive agents share 500 of them | |

## Decision methods and their names {#methods}

Every chapter uses these names. The plain name comes first; the abbreviation follows in brackets the first time.

| Plain name | Abbreviation | What it does |
| --- | --- | --- |
| First come, first served | FIFO | the job that arrived first |
| Shortest job first | SPT | the shortest processing time |
| Most urgent first | EDD | the earliest due date |
| Shortest setup first | SST | the job needing the shortest changeover, ties by earliest due date ("avoid colour changes" in the paint example) |
| Self-learning agent | DQN | a Double DQN that picks one of the four rules at each decision |
| Quick re-planning | Greedy replan | the re-planning loop with a greedy plan, no solver (an ablation) |
| Optimiser re-planning | CP-SAT | the re-planning loop with the CP-SAT model at every event |

Routing (which machine a job goes to) is the same for the four rules and the agent: the eligible machine with the
earliest expected finish.

## Reinforcement learning {#rl}

| Symbol | Meaning |
| --- | --- |
| $s_k$, $a_k$, $r_{k+1}$ | state, action and reward at decision $k$, taken at clock time $t_k$ |
| $a_k \in \{\text{FIFO}, \text{SPT}, \text{EDD}, \text{SST}\}$ | the action: which rule picks the next job |
| $\gamma = 0.97$ | discount factor |
| $Q(s, a)$, $Q_\theta$ | action value; the network with weights $\theta$ that approximates it |

## Statistics {#stats}

| Symbol | Meaning |
| --- | --- |
| $\Delta_i$ | paired difference on test day $i$, e.g. $\bar T^{\text{CP-SAT}}_i - \bar T^{\text{SST}}_i$ |
| $B = 10{,}000$ | bootstrap resamples; intervals are 95 % percentile intervals |

## Local symbols {#local}

Symbols that one or a few chapters introduce for their own argument. Each has one meaning across the course; the
last column says where it is used.

| Symbol | Meaning | Chapters |
| --- | --- | --- |
| $\sigma$ | an order (sequence) of jobs on one machine, wherever a sequence appears ($\sigma_m$ on machine $m$) | [[a01]], [[a02]], [[a03]], [[c02]] |
| $\theta$ | tightness of the due dates in the random single-machine instances (not $\tau$, the travel time) | [[a02]] |
| $\mathcal S$ | a selection of the disjunctive graph: one orientation for each pair of operations on a machine | [[a03]] |
| $\operatorname{len}(u, v)$ | length of a longest path from $u$ to $v$ in the disjunctive graph (not $\ell$, the transport lag) | [[a03]] |
| $\mu(j, k)$ | the routing: the machine assigned to operation $(j,k)$ | [[a01]], [[a03]], [[c01]], [[c03]] |
| $\alpha_{jk}$ | arrival of job $j$ at the machine of operation $(j,k)$ (not $A$, the availability) | [[a01]], [[a03]], [[c01]], [[c03]] |
| $R_{jm}$ | ready time of job $j$ at machine $m$ in a snapshot | [[a01]], [[b02]], [[c01]], [[c03]] |
| $\ell(m', m)$ | transport lag from machine $m'$ to machine $m$; $\tau(a, b)$ stays the travel time between stations | [[a01]], [[c01]], [[c03]] |
| $L = \lambda \bar F$ | Little's law, $\bar F$ the mean flow time (usually written $L = \lambda W$) | [[b01]] |
| $I_j(t)$ | priority index of a rule for job $j$ at time $t$ | [[b02]] |
| $\mathrm{slack}_j(t)$ | slack of job $j$ at time $t$, $d_j - t - P_j(t)$, with $P_j(t)$ its remaining fastest-route work; on one machine it is the ATCS slack $d_j - p_j - t$ | [[b02]], [[d01]] |
| $\hat C_{jm}$ | expected completion of job $j$ if it is routed to machine $m$ | [[b02]] |
| $U_m$ | work already committed to machine $m$ | [[b02]] |
| $\Sigma_t$, $\Pi_t$ | the snapshot of the shop at time $t$; the plan computed from it | [[c03]] |
| $\pi$ | a policy: a map from the state to an action | [[d01]] |
| $\Theta(t)$ | committed tardiness at time $t$ (not $E$, an end time) | [[d01]] |
| $W_f$ | fastest-route work of family $f$, the only meaning of $W$ in the course | [[a03]], [[b01]], [[d01]], [[d02]] |

Standard symbols keep their textbook meaning inside the derivation that uses them: the M/M/1 queue of [[b02]]
writes $\mu$ for the service rate, $N(t)$ for the number of jobs inside and $\pi_n$ for its stationary
distribution; the statistics of [[d01]] and [[d02]] write $\mu$, $\sigma$ and $\rho$ for a mean, a standard
deviation and a correlation; [[d01]] writes $\theta$ for the network's weights, $\beta$ for the soft-update rate
and $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ for the temporal-difference (TD) error, not the cart
allowance $\delta = 2$; Graham's notation in [[a01]] and [[a02]] writes $\alpha \mid \beta \mid \gamma$ for its
three fields.

A few other letters are reused on purpose, each inside one argument: [[a01]] writes $U_j$ for the late indicator
($1$ if $C_j > d_j$, else $0$) and [[b02]] writes $U_m$ for the work committed to machine $m$; in [[b01]], $N(t)$
is the number of jobs in the shop in Little's law ($N_s(t)$ the number of busy machines), $\omega_j$ the work
actually done on job $j$, changeovers included (at least $W_{f(j)}$), and $X^{(f)}$ the wait for the first
family-$f$ order of a split Poisson stream; [[c03]] writes $\eta$ for a safety margin, the minutes by which a plan
finishes an order before its due date.

## The worked example: one paint machine {#paint}

Introduced in chapter A1 and used as the small example of almost every chapter. One machine paints parts; it is set up for **blue**; any
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

Routes (the code's `routings`): for each family, the four operations and, for each, the eligible machines with
their processing times in minutes. Choosing one of them for each operation is the routing $\mu$.

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
