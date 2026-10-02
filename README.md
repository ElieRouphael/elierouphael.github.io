# elie-rouphael

Personal portfolio site for Elie Rouphael, built with [Astro](https://astro.build/).

## Project structure

```
.
├── public/                  # static assets served as-is
│   ├── assets/              # logo, profile photo, social icon SVGs
│   ├── images/              # gallery photos
│   └── tutorials/           # self-contained tutorial sites, one folder each
└── src/
    ├── components/          # Header, Footer, ItemCard, GalleryItem,
    │                        #   Lightbox, ResumeEntry, TutorialCard
    ├── data/                # research, reads, photos, resume, tutorials (TS)
    ├── layouts/             # BaseLayout.astro
    ├── pages/               # index, research, tutorials, resume,
    │                        #   suggested-reads, photography
    └── styles/              # global.css
```

Page content for the research papers, tutorials, suggested reads, gallery, and
resume lives in [`src/data/`](./src/data) as typed TypeScript modules. Edit those
files to update the corresponding pages.

### Adding a tutorial

1. Put the tutorial's static files in `public/tutorials/<slug>/` (with an
   `index.html`), or create an Astro page under `src/pages/tutorials/`.
2. Add an entry to [`src/data/tutorials.ts`](./src/data/tutorials.ts) with
   `href: '/tutorials/<slug>/'` and, optionally, a `cover` image.

The `/tutorials` page lists every entry in that array.

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
