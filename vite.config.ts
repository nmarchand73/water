import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

// Default `/` for `npm run dev` and local preview. Production GitHub Pages builds must pass
// `--base /<repo>/` (this repo: `/water/`) so `import.meta.env.BASE_URL` resolves textures/cubemap.
export default defineConfig({
  base: '/',
  plugins: [
    glsl({
      include: ['**/*.wgsl', '**/*.vert', '**/*.frag'],
      warnDuplicatedImports: true,
    }),
  ],
});
