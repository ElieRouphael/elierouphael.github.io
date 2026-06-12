export interface ReadItem {
  title: string;
  description: string;
  tag: string;
  author: string;
  source: string;
  href?: string;
}

export const reads: ReadItem[] = [
  {
    title: 'Feedback Systems: An Introduction for Scientists and Engineers',
    description:
      "\u00c5str\u00f6m & Murray's landmark textbook elegantly unifies classical and modern control, making the mathematics of stability, robustness, and performance accessible without sacrificing rigour. An essential companion for anyone entering the field.",
    tag: 'Textbook',
    author: '\u00c5str\u00f6m & Murray',
    source: 'Princeton UP, 2021',
  },
  {
    title: 'The C Programming Language',
    description:
      "Kernighan and Ritchie's canonical text remains the clearest exposition of C ever written. Every embedded and systems programmer should read it cover-to-cover at least once \u2014 ideally twice.",
    tag: 'Textbook',
    author: 'Kernighan & Ritchie',
    source: 'Prentice Hall, 1988',
  },
  {
    title: 'Introduction to Embedded Systems: A Cyber-Physical Systems Approach',
    description:
      "Lee & Seshia's freely available text bridges the gap between software and physical reality, covering concurrency, real-time semantics, and model-based design in a unified framework. Indispensable for CPS researchers and practitioners alike.",
    tag: 'Textbook',
    author: 'Lee & Seshia',
    source: 'MIT Press, 2017',
  },
  {
    title: 'Probabilistic Robotics',
    description:
      'Thrun, Burgard, and Fox set the standard for probabilistic state estimation in autonomous systems. From Kalman filters to particle filters and SLAM, the treatment is both rigorous and richly illustrated with real robot deployments.',
    tag: 'Textbook',
    author: 'Thrun, Burgard & Fox',
    source: 'MIT Press, 2005',
  },
  {
    title: 'A Mathematical Introduction to Logic',
    description:
      "Enderton's concise and precise treatment of first-order logic and computability underpins much of formal verification. I revisit certain chapters regularly when reasoning about program correctness and model-checking algorithms.",
    tag: 'Textbook',
    author: 'Herbert B. Enderton',
    source: 'Academic Press, 2001',
  },
  {
    title: 'The Art of Doing Science and Engineering: Learning to Learn',
    description:
      "Richard Hamming's transcribed lectures are a meditation on what it means to do first-rate research. Unconventional, opinionated, and frequently brilliant \u2014 a book that rewards rereading at every stage of a research career.",
    tag: 'Essay / Lecture Notes',
    author: 'Richard Hamming',
    source: 'Stripe Press, 2020',
  },
  {
    title: 'Thinking, Fast and Slow',
    description:
      "Kahneman's synthesis of decades of behavioural research offers a sobering corrective to overconfidence in engineering judgement. Understanding cognitive biases is, I believe, as important for an engineer as understanding circuit theory.",
    tag: 'Non-Fiction',
    author: 'Daniel Kahneman',
    source: 'Farrar, Straus & Giroux, 2011',
  },
  {
    title: 'The Soul of a New Machine',
    description:
      "Tracy Kidder's Pulitzer-winning account of the design of a new minicomputer captures the human drama of engineering under pressure better than any textbook. A reminder that behind every system there are people making difficult trade-offs under imperfect information.",
    tag: 'Non-Fiction',
    author: 'Tracy Kidder',
    source: 'Little, Brown, 1981',
  },
];
