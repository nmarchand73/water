import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

// Default `/` so `npm run dev`, `vite preview`, and local static servers resolve
// `import.meta.env.BASE_URL` + assets correctly. GitHub Pages uses `npm run deploy`
// which passes `--base /webgpu-water/` on the CLI (overrides this).
export default defineConfig({
  base: '/',
  plugins: [
    glsl({
      include: ['**/*.wgsl', '**/*.vert', '**/*.frag'],
      warnDuplicatedImports: true,
    }),
  ],
});
