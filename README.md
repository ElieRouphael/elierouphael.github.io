# elie-rouphael

A small static portfolio site built with plain HTML, CSS and JavaScript.

## Run locally

Pick one of the following methods to serve the site on your machine:

- Python 3 (no extra installs):

	```bash
	python3 -m http.server 8000
	```

	Open http://localhost:8000 in your browser.

- Node (using npx http-server):

	```bash
	npx http-server -c-1 -p 8080
	```

	Or if you'd like to use the npm script added to this repo:

	```bash
	npm run start
	```

	Open http://localhost:8080 in your browser.

## Hosting

This is a static site and can be hosted on any static hosting provider (GitHub Pages, Netlify, Vercel, Surge, etc.). Recommended options:

- GitHub Pages: push this repository to GitHub and enable Pages from repository settings. Serve from the `main` (or `master`) branch root.

- Netlify / Vercel: connect your GitHub repo and deploy — these providers auto-detect static sites.

## Notes

- All asset paths are relative so the site is ready for static hosting.
- The `package.json` includes quick start scripts; no build step is required.

## Files

- [index.html](index.html) — main page
- [styles.css](styles.css) — styles
- [script.js](script.js) — client-side behavior

If you want, I can add an automated GitHub Actions workflow (or a `gh-pages` deploy script) to publish to GitHub Pages.
