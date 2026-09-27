/* Text for the page that depends on the results. Every number is read from
   data.js, so re-running the experiments and export_web_data.py keeps the
   figures on the page correct. The interpretive sentences were written for
   the published results and should be re-read if the data changes. */
window.DQN_TEXT = (D, forward) => {
  const V = Object.fromEntries(D.variants.map(v => [v.name, v]));
  const f0 = x => Number(x).toFixed(0);
  const pct = x => `${Math.round(x * 100)}%`;
  const withCI = (mean, ci) => `${f0(mean)} (95% CI ${f0(ci[0])}–${f0(ci[1])})`;
  const FM = D.failure_modes || {};
  const T = {};

  // 4.1 Exploration
  const eps = V.baseline_fixed.epsilon_curve;
  T.eps_end_baseline = pct(eps[eps.length - 1]);

  // 2.7 Evidence that the terminal flag matters when the time limit is hit often
  const S = D.truncation_study;
  if (S) {
    T.trunc_evidence = `Does it matter in practice? When training rarely reaches the time limit, hardly at all, because the flag is almost never set. To test a case where it does, I shortened the limit to ${S.time_limit} steps, so a decent agent hits it constantly, and trained ${S.episodes} episodes with 10 seeds each way. Storing <code>terminated</code> alone gave a final greedy return of ${withCI(S.correct.final_mean, S.correct.final_ci)}, against ${withCI(S.wrong.final_mean, S.wrong.final_ci)} when truncation was stored as terminal. With 10 seeds the intervals only just separate, but the direction matches the theory.`;
  }

  // 4.2 Seeds
  const bf = V.baseline_fixed;
  const lo = Math.min(...bf.best_scores), hi = Math.max(...bf.best_scores);
  T.spag_caption = `<b>Read this before trusting any single RL curve.</b> The ten seeds of one configuration reach anywhere from ${f0(lo)} to ${f0(hi)} at their best checkpoint, and each rises and collapses at a different time. A difference of a few hundred points between two single runs says almost nothing about the configurations behind them.`;

  // 4.3 Comparison
  const means = D.variants.map(v => v.final_mean);
  const nPerfect = D.variants.reduce((a, v) => a + Math.round(v.final_frac_perfect * v.final_scores.length), 0);
  const nRuns = D.variants.reduce((a, v) => a + v.final_scores.length, 0);
  const avgBest = D.variants.reduce((a, v) => a + v.best_mean, 0) / D.variants.length;
  T.dot_caption = `<b>Final network:</b> every configuration averages between ${f0(Math.min(...means))} and ${f0(Math.max(...means))}, all the confidence intervals overlap, and ${nPerfect} of ${nRuns} final networks are perfect. <b>Best validation checkpoint:</b> flip the toggle. Every configuration roughly doubles (average ${f0(avgBest)}), and some seeds become perfect. Picking the checkpoint well matters more here than any single change to the algorithm.`;

  const slowEps = V.slow_exploration.epsilon_curve;
  T.results_prose = `
    <p>Three lessons from this comparison.</p>
    <p><b>No single change clearly helps within 300 episodes.</b> A deeper network, Double DQN, slower or faster exploration and per-step target syncing all give final returns whose intervals overlap the defaults'. With 10 seeds the fair summary is "no evidence of a difference", which is not the same as "no difference". Detecting a small effect would take more seeds or a longer training budget.</p>
    <p><b>The improved recipe is not better at this budget.</b> Double DQN, Huber loss, soft target updates and a lower learning rate are standard for good reasons, but they mostly pay off over longer training. Here the recipe finishes at ${f0(V.improved_dqn.final_mean)} against ${f0(V.baseline_fixed.final_mean)} for the defaults. I fixed the recipe before running anything, rather than tuning it until it won: tuning against the test results would make the comparison meaningless.</p>
    <p><b>The configuration that explores most does least badly.</b> Slow exploration still acts randomly about ${pct(slowEps[slowEps.length - 1])} of the time at the end, so its replay buffer stays full of varied situations, including the cart near the edge of the track. That fits section 5.5: its final networks drive off the track in ${FM.slow_exploration ? FM.slow_exploration.final.cart_off_track : "–"} of 100 test episodes, against ${FM.baseline_fixed ? FM.baseline_fixed.final.cart_off_track : "–"} for the defaults. Treat this as a hypothesis, though: its interval still overlaps the others.</p>`;

  // 4.4 Forgetting
  const peaks = D.variants.map(v => {
    const g = v.greedy_curve.mean.filter(y => y != null);
    return { peak: Math.max(...g), last: g[g.length - 1] };
  });
  const avgPeak = peaks.reduce((a, p) => a + p.peak, 0) / peaks.length;
  const avgLast = peaks.reduce((a, p) => a + p.last, 0) / peaks.length;
  T.curve_caption = `Averaged over seeds, the curves climb to around ${f0(avgPeak)} and then sag to around ${f0(avgLast)} by episode 300, and single seeds swing far more (section 5.2). This rise and fall is DQN's well-known instability, sometimes called catastrophic forgetting. Once the agent is good, the replay buffer fills with experience of balanced states, the network stops being reminded why the edges of the state space are dangerous, and the policy degrades. This is why keeping the best validation checkpoint pays off so much.`;

  // 4.5 Failure modes
  if (FM.baseline_fixed) {
    const f = FM.baseline_fixed.final;
    const tot = f.cart_off_track + f.pole_fell + f.time_limit;
    T.fail_caption = `<b>The pole almost never falls.</b> With the default settings, ${f.cart_off_track} of ${tot} test episodes of the final networks end with the cart driving off the track while the pole is still upright. The network learned the fast skill (keep the pole vertical) but not the slow one (stay near the centre). Drifting off takes a hundred steps or more, so the penalty arrives near the edge of the discount horizon from section 3.3, long after the decisions that caused it. Choose "DQN at the end of training" in the demo at the top to watch it happen. Exercise 4 suggests a fix.`;
  }

  // 4.6 Q-values
  const net = D.networks && D.networks.improved_best;
  if (net && forward) {
    const q = forward(net, [0, 0, 0, 0]);
    T.q_text = `Section 3.3 predicts that a well-balanced state is worth about <span class="m">1/(1-\\gamma) = 100</span>. The trained network in the demo, which balances perfectly, outputs only about ${q[0].toFixed(0)} for both actions at the centre. Nothing is broken. Each gradient step moves an estimate toward a target built from the next state's estimate, so value information travels backwards roughly one step per round of updates, and from a start near zero the estimates approach 100 slowly. Acting only needs the <em>ranking</em> of the two actions to be right, and it is right long before the values themselves converge. Exercise 1 explores this further.`;
  }

  T.compute_time = "about 4 CPU-hours, spread over 12 cores of a laptop";
  return T;
};
