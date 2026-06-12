export interface Photo {
  src: string;
  alt: string;
  caption: string;
}

export const photos: Photo[] = [
  { src: '/images/photo1.svg', alt: 'Lille, France', caption: 'Lille, France \u2014 2024' },
  { src: '/images/photo2.svg', alt: 'Architecture', caption: 'Architecture \u2014 Brussels, 2023' },
  { src: '/images/photo3.svg', alt: 'Nature', caption: 'Nature \u2014 Ardennes, 2023' },
  { src: '/images/photo4.svg', alt: 'Cityscape', caption: 'Cityscape \u2014 Paris, 2022' },
  { src: '/images/photo5.svg', alt: 'Portraits', caption: 'Portraits \u2014 Lille, 2022' },
  { src: '/images/photo6.svg', alt: 'Abstract', caption: 'Abstract \u2014 Studio, 2021' },
];
