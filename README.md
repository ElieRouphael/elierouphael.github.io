# elie-rouphael

Personal portfolio site for Elie Rouphael, built with [Astro](https://astro.build/).

## Project structure

```
.
├── public/                  # static assets served as-is
│   ├── assets/              # logo, profile photo, social icon SVGs
│   └── images/              # gallery photos
└── src/
    ├── components/          # Header, Footer, ItemCard, GalleryItem,
    │                        #   Lightbox, ResumeEntry
    ├── data/                # research, reads, photos, resume (TS)
    ├── layouts/             # BaseLayout.astro
    ├── pages/               # index, research, resume,
    │                        #   suggested-reads, photography
    └── styles/              # global.css
```

Page content for the research papers, suggested reads, gallery, and resume
lives in [`src/data/`](./src/data) as typed TypeScript modules. Edit those
files to update the corresponding pages.

## Commands

| Command           | Action                                        |
| :---------------- | :-------------------------------------------- |
| `npm install`     | Install dependencies                          |
| `npm run dev`     | Start dev server at `localhost:4321`          |
| `npm run build`   | Build the production site to `./dist/`        |
| `npm run preview` | Preview the production build locally          |
| `npm run astro`   | Run Astro CLI commands (`astro add`, `check`) |

## Deploying

The production build in `dist/` is fully static and can be deployed to any
static host (GitHub Pages, Netlify, Vercel, Cloudflare Pages, etc.).
