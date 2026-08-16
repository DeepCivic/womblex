// Theme + density are shell-level user preferences (DESIGN.md "Density and
// layout"): persisted locally, applied via `data-theme` / `data-density`
// attributes on the shell — not per-component props.

export type Theme = 'dark' | 'light';
export type Density = 'comfortable' | 'default' | 'compact';

const THEME_KEY = 'womblex-console:theme';
const DENSITY_KEY = 'womblex-console:density';

function readStored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
	if (typeof localStorage === 'undefined') return fallback;
	const stored = localStorage.getItem(key);
	return (allowed as readonly string[]).includes(stored ?? '') ? (stored as T) : fallback;
}

class Preferences {
	// Dark-first per DESIGN.md's adaptation principle 1.
	theme: Theme = $state(readStored(THEME_KEY, ['dark', 'light'] as const, 'dark'));
	density: Density = $state(
		readStored(DENSITY_KEY, ['comfortable', 'default', 'compact'] as const, 'default')
	);

	toggleTheme(): void {
		this.theme = this.theme === 'dark' ? 'light' : 'dark';
	}

	setDensity(density: Density): void {
		this.density = density;
	}
}

export const preferences = new Preferences();

// Effects that touch localStorage/DOM belong in the shell component
// (`+layout.svelte`), which runs only in the browser (`ssr = false`).
export function persistPreferences(): void {
	localStorage.setItem(THEME_KEY, preferences.theme);
	localStorage.setItem(DENSITY_KEY, preferences.density);
}
