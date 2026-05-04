# WebGPU Water

Real-time water simulation in the browser using **WebGPU**: reflections, refractions, caustics, and a heightfield wave simulation.

**Live demo:** [nmarchand73.github.io/water](https://nmarchand73.github.io/water/)

---

## About this fork

This repository ([nmarchand73/water](https://github.com/nmarchand73/water)) extends the WebGPU port by [jeantimex](https://github.com/jeantimex/webgpu-water), based on [Evan Wallace’s original WebGL Water](https://madebyevan.com/webgl-water/).

**Upstream (sync source):** [jeantimex/webgpu-water](https://github.com/jeantimex/webgpu-water)

Changes here include interaction and build fixes (pointer → ray mapping, WGSL `#include` via Vite), pool/world height scaling, **gravity on by default**, a collapsible help panel with full scrollable text, and **GitHub Pages** deployment with base path `/water/`.

---

## Requirements

- A browser with **WebGPU** enabled (recent Chrome, Edge, or Safari Technology Preview where available).

---

## Local development

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173/`). Assets use `import.meta.env.BASE_URL`; the dev server uses `/`.

---

## GitHub Pages

The site is published at **`https://nmarchand73.github.io/water/`**, so the production base path must be **`/water/`** (repository name).

### Option A — GitHub Actions (recommended)

1. In the repo: **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main`; the workflow [.github/workflows/pages.yml](.github/workflows/pages.yml) runs `npm run build:pages` and deploys the `dist` folder.

First deploy may require approving the `github-pages` environment once.

### Option B — Manual deploy with `gh-pages`

```bash
npm run deploy
```

Uses `npm run build:pages` then publishes `dist` to the `gh-pages` branch. In **Settings → Pages**, choose **Deploy from a branch**, branch **`gh-pages`**, folder **`/ (root)`**.

---

## Scripts

| Script | Purpose |
| ------ | ------- |
| `npm run dev` | Dev server (`base: /`) |
| `npm run build` | Production build (`base: /` — use for non-GitHub hosting at domain root) |
| `npm run build:pages` | Build for `github.io/<repo>/` (`base: /water/`) |
| `npm run deploy` | `build:pages` + push `dist` to `gh-pages` |
| `npm run preview` | Preview last build locally |

---

## Controls

| Action | Input |
| ------ | ----- |
| Ripples | Click / drag on the water (not on the sphere) |
| Rotate camera | Drag empty space, or **right-click** drag |
| Zoom | Mouse wheel |
| Pause | **Space** |
| Move sphere | Drag the sphere |
| Gravity | **G** (on by default) |
| Light from camera | **L** (hold) or enable in Settings |

Use the **menu** button (top-right) to open the help panel.

---

## Features

- Heightfield simulation (256×256), finite-difference waves  
- Raytraced reflections & refractions, Fresnel, cubemap sky  
- Real-time caustics on the pool floor  
- Buoyant sphere, optional density, wave tuning in **Settings**

---

## Technical notes

- **WGSL** shaders are preprocessed with [vite-plugin-glsl](https://github.com/UstymUkhman/vite-plugin-glsl); do **not** import shader entrypoints with `?raw` or `#include` will reach the browser unchanged and WebGPU will fail to compile.
- Water height is scaled consistently with pool size for rendering (see shader `common/functions.wgsl` and scene constants).

---

## Credits

- Original WebGL demo: [Evan Wallace](https://madebyevan.com/)  
- WebGPU port (upstream): [jeantimex](https://github.com/jeantimex)  
- Fork maintenance: [nmarchand73](https://github.com/nmarchand73)

## References

- [Original WebGL Water](https://madebyevan.com/webgl-water/)  
- [Rendering realtime caustics in WebGL](https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c)  
- [WebGPU](https://www.w3.org/TR/webgpu/) · [WGSL](https://www.w3.org/TR/WGSL/)

## License

Open source; the original WebGL water demo is by Evan Wallace (see upstream and original site for terms).
