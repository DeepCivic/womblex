import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			preprocess: vitePreprocess(),
			// Kit 3 dropped the built-in `$lib` alias; keep it rather than
			// rewriting every import to `#lib`.
			alias: { $lib: 'src/lib' },
			// Static SPA build: the sidecar has no Node runtime (Dockerfile.ui),
			// so the whole app is prerendered/fallback-served as plain files and
			// FastAPI mounts `build/` (womblex/ui/app.py resolves it relative to
			// the repo root, matching where Dockerfile.ui's builder stage COPYs
			// the output).
			adapter: adapter({
				pages: 'build',
				assets: 'build',
				fallback: 'index.html',
				strict: false
			})
		})
	],
	server: {
		// Dev-only: the sidecar serves `/api/*` itself in every other
		// context. Point at a locally running `womblex ui` (default port
		// 8080) instead of a bundled mock backend.
		proxy: {
			'/api': 'http://127.0.0.1:8080'
		}
	}
});
