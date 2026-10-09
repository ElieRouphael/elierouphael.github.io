export interface Tutorial {
  title: string;
  description: string;
  /** Short topic labels shown under the description. */
  topics: string[];
  /** Assumed background, e.g. "Python, PyTorch". */
  prerequisites?: string;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  /** When the tutorial was added (ISO date-time). Used only to sort the list; never displayed. */
  added: string;
  /** URL of the tutorial. Static tutorials live in public/tutorials/<slug>/. */
  href: string;
  /** Optional preview image shown next to the card. */
  cover?: {
    src: string;
    alt: string;
  };
  /**
   * Set to true to hide this tutorial from the public site.
   * Draft tutorials are excluded from `npm run build` output entirely,
   * but still show up while running `npm run dev` so you can preview them.
   */
  draft?: boolean;
}

export const tutorials: Tutorial[] = [
  {
    title: 'Teaching a Network to Balance a Pole',
    description:
      'Reinforcement learning from scratch: what RL is, the Gymnasium API, the mathematics from Markov decision processes to Double DQN, a line-by-line PyTorch implementation, and what training really looks like across 70 runs. The trained network runs live in the browser.',
    topics: ['Reinforcement Learning', 'Deep Q-Networks', 'PyTorch', 'Gymnasium'],
    prerequisites: 'Python and PyTorch',
    level: 'Intermediate',
    added: '2026-09-27T12:00',
    href: '/tutorials/dqn-cartpole/index.html',
    cover: {
      src: '/tutorials/dqn-cartpole/cover.gif',
      alt: 'A trained DQN agent balancing a pole on a cart',
    },
  },
  {
    title: 'Trustworthy AI Perception in a Control Loop',
    description:
      'A ten-chapter course that builds a small lane-keeping system which stays safe when its AI perception fails: the vehicle model, LQR control, a synthetic camera, deep ensembles, the Kalman filter and the NIS consistency test, integrity monitoring and shared control with a human driver, all tested in a closed-loop capstone. Every chapter is a runnable notebook.',
    topics: ['Control', 'Kalman Filtering', 'Uncertainty', 'Shared Control', 'NumPy'],
    prerequisites: 'Linear algebra, probability and Python',
    level: 'Advanced',
    added: '2026-09-28T16:15',
    href: '/tutorials/trustdrive/index.html',
    cover: {
      src: '/tutorials/trustdrive/cover.png',
      alt: 'The network reports flat uncertainty while a model-based consistency check spikes once the perception goes wrong',
    },
  },
  {
    title: 'Bayesian Statistics from Scratch',
    description:
      "A ten-chapter course from Bayes' theorem to Markov chain Monte Carlo: probability and distributions, likelihood and confidence intervals, priors and posteriors, conjugate models, choosing priors, Bayesian linear regression and a Metropolis sampler written from scratch. Every formula is derived by hand and checked in a runnable notebook.",
    topics: ['Bayesian Inference', 'Probability', 'Conjugate Priors', 'MCMC', 'SciPy'],
    prerequisites: 'Algebra, a little calculus and Python',
    level: 'Beginner',
    added: '2026-09-28T16:52',
    href: '/tutorials/bayesian-statistics/index.html',
    cover: {
      src: '/tutorials/bayesian-statistics/cover.png',
      alt: 'A grey prior and an orange likelihood combine into a narrower blue posterior between them',
    },
  },
  {
    title: 'Understanding Time Series',
    description:
      'An eleven-chapter course in time-series analysis and forecasting on one real dataset, six years of hourly air pollution in Beijing: time in pandas, decomposition, stationarity, honest evaluation against baselines, exponential smoothing, ARMA and SARIMAX with weather inputs, Prophet, machine learning and deep learning, ending with a final test on a year kept locked until the last chapter. Every chapter is a runnable notebook.',
    topics: ['Forecasting', 'ARIMA', 'Prophet', 'Machine Learning', 'pandas'],
    prerequisites: 'Python and a little pandas',
    level: 'Intermediate',
    added: '2026-09-29T18:43',
    href: '/tutorials/time-series/index.html',
    cover: {
      src: '/tutorials/time-series/cover.png',
      alt: 'Daily PM2.5 in Beijing through the end of 2015, repeatedly climbing above the dashed heavy-pollution line',
    },
  },
  {
    title: 'A Tour of Machine Learning',
    description:
      'Eleven standalone notebooks, one machine-learning idea each, every one on real data judged by its own honest result: linear regression, the calculus behind training, a from-scratch SVM decision boundary on iris flowers, a random forest against a single tree on heart-disease data, K-means vs. k-means++ on mall customers, a neural network written in plain NumPy, transfer learning on shifted digits, an LSTM against a naive baseline, an autoencoder that catches an anomaly it was never shown, attention worked out by hand, and federated learning simulated from scratch.',
    topics: ['Machine Learning', 'Deep Learning', 'scikit-learn', 'NumPy', 'TensorFlow/Keras'],
    prerequisites: 'Python and a little numpy/pandas; no prior ML background needed',
    level: 'Beginner',
    added: '2026-09-29T20:47',
    href: '/tutorials/ml-tour/index.html',
    cover: {
      src: '/tutorials/ml-tour/cover.png',
      alt: 'Five customer segments found by K-means, coloured by cluster, with an X marking each centroid',
    },
    draft: false,
  },
  {
    title: 'Learning to Identify New Dynamical Systems from a Few Experiments',
    description:
      "A nine-chapter research tutorial built from the author's own work on few-shot system identification: a family of nonlinear oscillators, output-error identification from scratch, pretraining a two-number shared context from 20 systems, few-shot Bayesian adaptation, a data-budget study, Bayesian-optimal experiment design, detecting when the shared knowledge no longer applies, and model-based control, closing with a map onto real foundation models for dynamical systems.",
    topics: ['System Identification', 'Dynamical Systems', 'Bayesian Inference', 'Experiment Design', 'NumPy'],
    prerequisites: 'Python, a little linear algebra and probability',
    level: 'Advanced',
    added: '2026-09-30T09:57',
    href: '/tutorials/few-shot-dynamical-systems/index.html',
    cover: {
      src: '/tutorials/few-shot-dynamical-systems/cover.png',
      alt: 'A cluster of training-system contexts with a posterior ellipse around a new system, estimated from two seconds of data, containing the true system',
    },
    draft: false,
  },
  {
    title: 'Advanced MPC and Learning MPC',
    description:
      'A ten-notebook course on constrained optimal control that never treats the optimizer as a black box: convex optimization and KKT conditions, invariant sets, recursive feasibility and stability of MPC, collision avoidance as a mixed-integer decision, Learning MPC from repeated executions, minimum-time racing and learned vehicle models, and the bridge from dynamic programming to reinforcement learning, closing with one design checked end to end.',
    topics: ['MPC', 'Learning MPC', 'Invariant Sets', 'Optimization', 'Dynamic Programming'],
    prerequisites: 'Linear algebra, basic control (state-space models, stability) and Python',
    level: 'Advanced',
    added: '2026-10-02T10:16',
    href: '/tutorials/mpc-lmpc-lab/index.html',
    cover: {
      src: '/tutorials/mpc-lmpc-lab/cover.png',
      alt: 'A constraint box containing the maximal control-invariant set and the smaller LQR terminal set, with an MPC trajectory converging to the origin',
    },
    draft: false,
  },
  {
    title: 'Finding a Lost Robot with a Particle Filter',
    description:
      'A beginner\'s tutorial on Monte Carlo localization: a robot with no GPS finds itself on a hilly map using only its wheels, a compass and an altimeter. A warm-up on a loop track, noisy motion, the bell curve, resampling with a comb, jitter and the kidnapped robot, one full step worked out by hand, then the whole filter as four short Python functions, experiments that show when it fails, a comparison with the Kalman filter and exercises with solutions. Every idea comes with a live demo in the browser, and optional boxes give the math behind it.',
    topics: ['Particle Filters', 'Localization', 'Robotics', 'Probability', 'NumPy'],
    prerequisites: 'None: no robotics, probability or programming background needed',
    level: 'Beginner',
    added: '2026-10-09T12:00',
    href: '/tutorials/particle-filter/index.html',
    cover: {
      src: '/tutorials/particle-filter/cover.jpg',
      alt: 'A hilly map with the robot in orange and its guesses in blue, gathered into several clumps that all fit the readings so far',
    },
    draft: false,
  },
];
