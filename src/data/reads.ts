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

// Entries are sorted by the first author's family name on the page, so order here doesn't matter.
export const reads: ReadSection[] = [
  {
    heading: 'Fiction',
    subsections: [
      {
        heading: 'Classics',
        entries: [
          { title: 'The Idiot', author: 'Fyodor Dostoevsky' },
          { title: 'The Brothers Karamazov', author: 'Fyodor Dostoevsky' },
          { title: 'Demons', author: 'Fyodor Dostoevsky' },
          { title: 'White Nights', author: 'Fyodor Dostoevsky' },
          { title: 'L’Étranger', author: 'Albert Camus' },
          { title: 'Fahrenheit 451', author: 'Ray Bradbury' },
          { title: 'Brave New World', author: 'Aldous Huxley' },
          { title: 'Nineteen Eighty-Four', author: 'George Orwell' },
          { title: 'Animal Farm', author: 'George Orwell' },
          { title: 'Do Androids Dream of Electric Sheep?', author: 'Philip K. Dick' },
          { title: 'Women', author: 'Charles Bukowski' },
          { title: 'The Alchemist', author: 'Paulo Coelho' },
          { title: 'Sherlock Holmes', author: 'Arthur Conan Doyle', note: 'the complete novels and stories, volumes I and II' },
          { title: 'The Complete Fiction', author: 'H. P. Lovecraft' },
        ],
      },
      {
        heading: 'Contemporary',
        entries: [
          { title: 'Fairy Tale', author: 'Stephen King' },
          { title: 'You Like It Darker', author: 'Stephen King' },
          { title: 'Small Great Things', author: 'Jodi Picoult' },
          { title: 'A Spark of Light', author: 'Jodi Picoult' },
          { title: 'Angels & Demons', author: 'Dan Brown' },
          { title: 'The Da Vinci Code', author: 'Dan Brown' },
          { title: 'A Time for Mercy', author: 'John Grisham' },
          { title: 'Transmetropolitan', author: 'Warren Ellis & Darick Robertson', note: 'graphic novel' },
        ],
      },
      {
        heading: 'French Novels',
        entries: [
          { title: 'Demain', author: 'Guillaume Musso' },
          { title: 'La Vie secrète des écrivains', author: 'Guillaume Musso' },
          { title: 'La Fille de Brooklyn', author: 'Guillaume Musso' },
          { title: 'Un appartement à Paris', author: 'Guillaume Musso' },
          { title: 'La vie est un roman', author: 'Guillaume Musso' },
          { title: 'L’Instant présent', author: 'Guillaume Musso' },
          { title: 'Parce que je t’aime', author: 'Guillaume Musso' },
          { title: 'Central Park', author: 'Guillaume Musso' },
          { title: 'La Jeune Fille et la Nuit', author: 'Guillaume Musso' },
          { title: 'Au soleil redouté', author: 'Michel Bussi' },
          { title: 'Un avion sans elle', author: 'Michel Bussi' },
          { title: 'Gravé dans le sable', author: 'Michel Bussi' },
          { title: 'On la trouvait plutôt jolie', author: 'Michel Bussi' },
          { title: 'Rien ne t’efface', author: 'Michel Bussi' },
          { title: 'C’est arrivé la nuit', author: 'Marc Levy' },
          { title: 'Vous revoir', author: 'Marc Levy' },
          { title: 'Ghost in Love', author: 'Marc Levy' },
          { title: 'La Dernière des Stanfield', author: 'Marc Levy' },
          { title: 'Elle & lui', author: 'Marc Levy' },
          { title: 'Et si c’était vrai…', author: 'Marc Levy' },
          { title: 'Les Derniers Jours de nos pères', author: 'Joël Dicker' },
          { title: 'Le Livre des Baltimore', author: 'Joël Dicker' },
          { title: 'La Vérité sur l’affaire Harry Quebert', author: 'Joël Dicker' },
          { title: 'L’Énigme de la chambre 622', author: 'Joël Dicker' },
          { title: 'Vertige', author: 'Franck Thilliez' },
          { title: 'Le Manuscrit inachevé', author: 'Franck Thilliez' },
          { title: 'Labyrinthes', author: 'Franck Thilliez' },
          { title: 'Il était deux fois', author: 'Franck Thilliez' },
          { title: 'Entre deux mondes', author: 'Olivier Norek' },
          { title: 'Trilogie 93', author: 'Olivier Norek' },
          { title: 'Surface', author: 'Olivier Norek' },
          { title: 'Que ta volonté soit faite', author: 'Maxime Chattam' },
          { title: 'Carnages', author: 'Maxime Chattam' },
          { title: 'De cauchemar et de feu', author: 'Nicolas Lebel' },
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
            title: 'Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches',
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
            author: 'Roland Tóth',
          },
          { title: 'Identification et commande des systèmes', author: 'Ioan Doré Landau' },
          { title: 'Algèbre et analyse pour l’automatique', author: 'Jean-Pierre Richard' },
          { title: 'Stable Adaptive Systems', author: 'Kumpati S. Narendra & Anuradha M. Annaswamy' },
        ],
      },
    ],
  },
  {
    heading: 'Self Development',
    entries: [
      { title: 'The Subtle Art of Not Giving a F*ck', author: 'Mark Manson' },
      { title: '12 Rules for Life', author: 'Jordan Peterson' },
      { title: 'We Who Wrestle with God', author: 'Jordan Peterson' },
      { title: 'How to Talk to Anyone', author: 'Leil Lowndes' },
      { title: 'The Mindful Body', author: 'Ellen Langer' },
      { title: 'The 7 Habits of Highly Effective People', author: 'Stephen R. Covey' },
      { title: 'Atomic Habits', author: 'James Clear' },
      { title: 'Influence', author: 'Robert B. Cialdini' },
      { title: 'The Black Swan', author: 'Nassim Nicholas Taleb' },
      { title: 'Blink', author: 'Malcolm Gladwell' },
      { title: 'David and Goliath', author: 'Malcolm Gladwell' },
    ],
  },
  {
    heading: 'Non-fiction',
    entries: [
      { title: 'Essays', author: 'George Orwell' },
      { title: 'Le Diable et Sherlock Holmes', author: 'David Grann' },
      { title: 'De l’inconvénient d’être né', author: 'Emil Cioran' },
    ],
  },
  {
    heading: 'Shorts',
    entries: [
      { title: 'The Fall of the House of Usher', author: 'Edgar Allan Poe' },
      { title: 'The Tell-Tale Heart', author: 'Edgar Allan Poe' },
      { title: 'The Black Cat', author: 'Edgar Allan Poe' },
      { title: 'The Facts in the Case of M. Valdemar', author: 'Edgar Allan Poe' },
      { title: 'Creepy Stories', author: 'Edgar Allan Poe', note: 'collection' },
      { title: 'The Shot', author: 'Alexander Pushkin' },
      { title: 'That Evening Sun', author: 'William Faulkner' },
      { title: 'Le Horla', author: 'Guy de Maupassant' },
      {
        title: 'Red Screen',
        author: 'Stephen King',
        note: 'from the collection “You Like It Darker”',
      },
      {
        title: 'The Sisters',
        author: 'James Joyce',
        note: 'from the collection “Dubliners”',
      },
      {
        title: 'An Encounter',
        author: 'James Joyce',
        note: 'from the collection “Dubliners”',
      },
    ],
  },
  {
    heading: 'Poems',
    entries: [
      { title: 'Lenore', author: 'Edgar Allan Poe' },
      { title: 'Annabel Lee', author: 'Edgar Allan Poe' },
      { title: 'The Raven', author: 'Edgar Allan Poe' },
      { title: 'Selected Poems', author: 'T. S. Eliot' },
      { title: 'Love Is a Dog from Hell', author: 'Charles Bukowski' },
    ],
  },
  {
    heading: 'Chess',
    entries: [
      { title: 'Jouez 1.e4 !', author: 'John Shaw', note: 'tomes 1 à 3' },
      { title: 'The Caro-Kann: Move by Move', author: 'Cyrus Lakdawala' },
      { title: 'The Closed Sicilian: Move by Move', author: 'Carsten Hansen' },
      { title: 'The Modernized Grünfeld Defense', author: 'Yaroslav Zherebukh' },
      { title: 'Bologan’s King’s Indian', author: 'Victor Bologan' },
      { title: 'Van Perlo’s Endgame Tactics', author: 'Ger van Perlo' },
      { title: 'The Woodpecker Method', author: 'Axel Smith & Hans Tikkanen' },
      { title: 'Chess Training, Volume 3: Legendary Games', author: 'Romain Édouard' },
      { title: '1001 Deadly Checkmates', author: 'John Nunn' },
      { title: 'Les 100 finales qu’il faut connaître', author: 'Jesús de la Villa' },
    ],
  },
  {
    heading: 'Spirituality',
    entries: [
      { title: 'Discours ascétiques', author: 'Saint Isaac le Syrien' },
      { title: 'Dieu et l’homme', author: 'Antoine Bloom' },
      { title: 'La Foi et le doute', author: 'Antoine Bloom' },
      { title: 'Vivre la communauté chrétienne', author: 'Antoine Bloom' },
      { title: 'Étapes de la vie spirituelle', author: 'Antoine Bloom' },
      { title: 'Initiation à la théologie byzantine', author: 'Jean Meyendorff' },
    ],
  },
  {
    heading: 'Art',
    entries: [
      {
        title: 'Van Gogh : L’Œuvre complet – Peinture',
        author: 'Ingo F. Walther & Rainer Metzger',
      },
    ],
  },
];
