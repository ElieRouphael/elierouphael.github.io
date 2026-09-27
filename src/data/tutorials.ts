export interface Tutorial {
  title: string;
  description: string;
  /** Short topic labels shown under the description. */
  topics: string[];
  /** Assumed background, e.g. "Python, PyTorch". */
  prerequisites?: string;
  year: string;
  /** URL of the tutorial. Static tutorials live in public/tutorials/<slug>/. */
  href: string;
  /** Optional preview image shown next to the card. */
  cover?: {
    src: string;
    alt: string;
  };
}

export const tutorials: Tutorial[] = [
  {
    title: 'Teaching a Network to Balance a Pole',
    description:
      'Reinforcement learning from scratch: what RL is, the Gymnasium API, the mathematics from Markov decision processes to Double DQN, a line-by-line PyTorch implementation, and what training really looks like across 70 runs. The trained network runs live in the browser.',
    topics: ['Reinforcement Learning', 'Deep Q-Networks', 'PyTorch', 'Gymnasium'],
    prerequisites: 'Python and PyTorch',
    year: '2026',
    href: '/tutorials/dqn-cartpole/',
    cover: {
      src: '/tutorials/dqn-cartpole/cover.gif',
      alt: 'A trained DQN agent balancing a pole on a cart',
    },
  },
];
