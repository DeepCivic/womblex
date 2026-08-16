import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
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
	}
};

export default config;
