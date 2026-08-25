<script lang="ts">
	import { preferences, persistPreferences, type Density } from '$lib/stores/preferences.svelte';

	const LEVELS: { value: Density; label: string }[] = [
		{ value: 'comfortable', label: 'Comfortable' },
		{ value: 'default', label: 'Default' },
		{ value: 'compact', label: 'Compact' }
	];

	function select(level: Density): void {
		preferences.setDensity(level);
		persistPreferences();
	}
</script>

<!--
	`aria-pressed` toggle buttons in a labelled group, not role="radiogroup".
	A radiogroup announces itself as one tab stop navigated by arrow keys;
	these are three ordinary tab stops, so declaring that role would promise
	the screen reader a keyboard contract the component does not honour.
-->
<div class="flex items-center rounded-md border border-border p-0.5 text-xs" role="group" aria-label="Row density">
	{#each LEVELS as level (level.value)}
		<button
			type="button"
			aria-pressed={preferences.density === level.value}
			class={[
				'rounded-sm px-2 py-1.5',
				'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
				preferences.density === level.value
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'
			]}
			onclick={() => select(level.value)}
		>
			{level.label}
		</button>
	{/each}
</div>
