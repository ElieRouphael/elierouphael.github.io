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
    'Hello! I am a Senior Lecturer at the University of Lille, where I specialize in a wide range of topics, including control theory, C programming, sequential logic, and autonomous systems, among others. My academic background includes a PhD in learning and computer science, where I focused on using AI tools to develop system identification algorithms based on realization theory. I also hold a Master\u2019s degree in electrical energy and control theory. My research and teaching aim to make complex ideas accessible, while pushing the boundaries of what\u2019s possible in these fields',
    'Outside of my academic work, I\u2019m an active chess player, enjoying the mental challenge and strategic thinking the game demands\u2014skills that also complement my approach to problem-solving in control systems.',
    'I also have an enthusiastic interest in photography, finding creativity in capturing the world through the lens. You may find some of my humble works here.',
    'On this website, you\u2019ll also find, as well, a collection of my recommended reads, ranging from fiction to essays to technical works. These are reads that have inspired me, and I hope they will provide the same value and insight to others.',
  ],
  tags: [
    'Control Theory',
    'C Programming',
    'Sequential Logic',
    'Autonomous Systems',
    'Robotics',
    'Embedded Systems',
  ],
};
