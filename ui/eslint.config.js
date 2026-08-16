import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import tseslint from 'typescript-eslint';
import globals from 'globals';

export default tseslint.config(
	js.configs.recommended,
	...tseslint.configs.recommended,
	...svelte.configs.recommended,
	{
		languageOptions: {
			globals: { ...globals.browser, ...globals.es2021 }
		}
	},
	{
		files: ['**/*.svelte', '**/*.svelte.ts'],
		languageOptions: {
			parserOptions: { parser: tseslint.parser, extraFileExtensions: ['.svelte'] }
		}
	},
	{
		rules: {
			// Static shell routes (docs/ui-plan.md merge 4) — plain `href`s, not
			// SvelteKit's typed-router `resolve()` wrapper.
			'svelte/no-navigation-without-resolve': 'off'
		}
	},
	{
		ignores: ['build/', '.svelte-kit/', 'node_modules/']
	}
);
