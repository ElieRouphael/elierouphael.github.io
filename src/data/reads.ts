export interface ReadEntry {
  title: string;
  author?: string;
  note?: string;
}

export interface ReadSubsection {
  heading: string;
  entries: ReadEntry[];
}

export interface ReadSection {
  heading: string;
  entries?: ReadEntry[];
  subsections?: ReadSubsection[];
}

export const reads: ReadSection[] = [
  {
    heading: 'Fiction',
    subsections: [
      {
        heading: 'Classics',
        entries: [
          { title: 'The Idiot', author: 'Fyodor Dostoevsky' },
        ],
      },
      {
        heading: 'Contemporary',
        entries: [
          { title: 'Fairy Tale', author: 'Stephen King' },
          { title: 'Small Great Things', author: 'Jodi Picoult' },
          { title: 'A Spark of Light', author: 'Jodi Picoult' },
        ],
      },
    ],
  },
  {
    heading: 'Science',
    subsections: [
      {
        heading: 'Control & System Identification',
        entries: [
          {
            title: 'Linear Parameter Varying Control: Theory and Application to Automotive Systems',
            author: 'Olivier Sename',
          },
          {
            title: 'Optimal State Estimation: Kalman, H\u221e, and Nonlinear Approaches',
            author: 'Dan Simon',
          },
          {
            title: 'Subspace Methods for System Identification',
            author: 'Tohru Katayama',
          },
          {
            title: 'Control and System Theory',
            author: 'Jan H. Van Schuppen',
          },
          {
            title: 'Modeling and Identification of Linear Parameter-Varying Systems',
            author: 'Roland T\u00f3th',
          },
        ],
      },
    ],
  },
  {
    heading: 'Self Development',
    entries: [
      { title: 'The Subtle Art of Not Giving a F*ck', author: 'Mark Manson' },
      { title: '12 Rules for Life', author: 'Jordan Peterson' },
      { title: 'How to Talk to Anyone', author: 'Leil Lowndes' },
      { title: 'The Mindful Body', author: 'Ellen Langer' },
    ],
  },
  {
    heading: 'Shorts',
    entries: [
      { title: 'The Fall of the House of Usher', author: 'Edgar Allan Poe' },
      { title: 'The Tell-Tale Heart', author: 'Edgar Allan Poe' },
      { title: 'The Black Cat', author: 'Edgar Allan Poe' },
      { title: 'The Facts in the Case of M. Valdemar', author: 'Edgar Allan Poe' },
      { title: 'The Shot', author: 'Alexander Pushkin' },
      { title: 'That Evening Sun', author: 'William Faulkner' },
      {
        title: 'Red Screen',
        author: 'Stephen King',
        note: 'from the collection \u201cYou Like It Darker\u201d',
      },
      {
        title: 'The Sisters',
        author: 'James Joyce',
        note: 'from the collection \u201cDubliners\u201d',
      },
      {
        title: 'An Encounter',
        author: 'James Joyce',
        note: 'from the collection \u201cDubliners\u201d',
      },
    ],
  },
  {
    heading: 'Poems',
    entries: [
      { title: 'Lenore', author: 'Edgar Allan Poe' },
      { title: 'Annabel Lee', author: 'Edgar Allan Poe' },
      { title: 'The Raven', author: 'Edgar Allan Poe' },
    ],
  },
];
