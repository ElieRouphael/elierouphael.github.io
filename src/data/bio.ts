export interface BioImage {
  src: string;
  alt: string;
}

export interface Bio {
  name: string;
  subtitle: string;
  image: BioImage;
  paragraphs: string[];
  tags: string[];
}

export const bio: Bio = {
  name: 'Elie Rouphael',
  subtitle: 'Postdoctoral Researcher in AI, System Identification & Control',
  image: {
    src: '/assets/elie-img-1.jpg',
    alt: 'Elie Rouphael',
  },
  paragraphs: [
    'Hello! I am a Postdoctoral Researcher at LabCom I-TireLab, a joint lab between GIPSA-lab (Grenoble INP \u2013 UGA), LIAS (University of Poitiers) and Michelin, where I develop physics-informed and data-driven models for tire-road interaction and vehicle dynamics. I hold a PhD in Automatic Control and Computer Science from the University of Lille (CRIStAL Laboratory), where I built a stochastic realization theory for switched and LPV systems, and a Master\u2019s in Automatic Control and Electrical Energy from the University of Poitiers. Alongside my research, I\u2019ve taught control theory, programming, and embedded systems at the University of Lille. My work aims to make complex dynamical systems easier to understand, while pushing the boundaries of what\u2019s possible in system identification and learning-based modeling.',
    'Outside of my research, I\u2019m an active chess player, enjoying the mental challenge and strategic thinking the game demands\u2014skills that also complement my approach to problem-solving in control systems.',
    'I also have an enthusiastic interest in photography, finding creativity in capturing the world through the lens. You may find some of my humble works here.',
    'On this website, you\u2019ll also find a collection of my recommended reads, ranging from fiction to essays to technical works. These are reads that have inspired me, and I hope they will provide the same value and insight to others.',
  ],
  tags: [
    'System Identification',
    'Control Theory',
    'Hybrid & Switched Systems',
    'Machine Learning',
    'Vehicle Dynamics',
    'Embedded Systems',
  ],
};
