export interface Photo {
  src: string;
  alt: string;
  caption: string;
}

export const photos: Photo[] = [
  { src: '/images/lille-1.jpg', alt: 'Lille, France', caption: 'Lille, France - 2024' },
  { src: '/images/lyon-1.jpg', alt: 'Lyon, France', caption: 'Lyon, France - 2023' },
  { src: '/images/paris-1.jpg', alt: 'Paris, France', caption: 'Paris, France - 2023' },
  { src: '/images/lyon-2.jpg', alt: 'Lyon, France', caption: 'Lyon, France - 2022' },
  { src: '/images/lyon-3.jpg', alt: 'Lyon, France', caption: 'Lyon, France - 2022' },
  { src: '/images/lyon-4.jpg', alt: 'Lyon, France', caption: 'Lyon, France - 2021' },
];
