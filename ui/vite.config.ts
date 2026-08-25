import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		// Dev-only: the sidecar serves `/api/*` itself in every other
		// context. Point at a locally running `womblex ui` (default port
		// 8080) instead of a bundled mock backend.
		proxy: {
			'/api': 'http://127.0.0.1:8080'
		}
	}
});
