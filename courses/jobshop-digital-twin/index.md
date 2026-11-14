---
eyebrow: A ten-chapter course · scheduling, simulation, constraint programming, reinforcement learning
lede: >-
  A workshop must keep deciding which order each machine works on next, while orders arrive at random, a change of
  product costs set-up time, and machines break down. This course teaches how to answer that question well, from the
  first definitions to a working system: how to describe a schedule mathematically and measure it, when a simple
  rule is provably optimal and why it often cannot be, how to simulate a shop you cannot experiment on, how to write
  the shop as a constraint-programming model and let a solver plan it, how a learning agent fares at the same task,
  and how to decide, with statistics, which method really is better.
assumes: No background in scheduling needed · Python, basic probability and linear inequalities help
description: >-
  A ten-chapter course on dynamic scheduling, built on an open-source simulated workshop: schedules and their
  measures, optimal and non-optimal rules, discrete-event simulation, a constraint-programming model solved with
  CP-SAT, rescheduling on a rolling horizon, a reinforcement-learning dispatcher, and a fair benchmark.
---

## What you will be able to do {#goals}

1. **Write a scheduling problem down precisely**: jobs, operations, machines, constraints and an objective, in the
   standard notation, and compute every measure of a schedule by hand. Chapters [[a01]] and [[a03]].
2. **Know when a simple rule is the best possible** (and prove it with an exchange argument), and recognise the
   problems where no rule can be, such as total tardiness with changeovers; then use rules as priority indices that
   dispatch a running shop. Chapters [[a02]] and [[b02]].
3. **Build a discrete-event simulation** of a shop with random arrivals and breakdowns, and predict its load and
   availability before running it. Chapter [[b01]].
4. **Model a flexible job shop for a constraint-programming solver**, and explain how the solver finds a plan and
   proves that no better plan exists. Chapters [[c01]] and [[c02]].
5. **Turn a planner into a controller** that re-plans at every disruption, and know what such a controller can and
   cannot see. Chapter [[c03]].
6. **Formulate dispatching as a reinforcement-learning problem**, train an agent, and understand why it may not
   beat a good rule. Chapter [[d01]].
7. **Run a benchmark that deserves to be believed**: same random days for every method, paired comparisons,
   bootstrap intervals, untouched test data and an independent checker. Chapter [[d02]], with the results of
   [[b02]], [[c03]] and [[d01]].

## How the chapters are built {#how}

Each chapter teaches one idea. It starts in plain words, with a small example, before the precise definition and
the mathematics; it has a live demo that runs in the browser; it reads the lines of the
[repository](https://github.com/ElieRouphael/jobshop-digital-twin) that implement the idea; it ends with questions
and exercises whose solutions are folded underneath; and every number on the page is printed by a script you can
run. The whole course runs on one synthetic workshop, described once on the [notation page](notation.html):
five machines, two carts, four product families, sixty orders a simulated day.

## Where the course ends up {#preview}

The last chapters compare seven ways of running the workshop on the same twenty simulated days. Re-planning the
whole shop with a constraint-programming solver at every disruption cuts the mean tardiness by 43 % under frequent
breakdowns, against the best rule; most of that gain comes from re-planning itself, and the learning agent does no
better than the best rule. Chapters [[c03]], [[d01]] and [[d02]] explain why, and how much of that conclusion the
evidence supports.

![Mean tardiness of the seven methods on the twenty test days, with and without breakdowns](tardiness.png)
