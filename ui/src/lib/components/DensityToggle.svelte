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

<div class="flex items-center rounded-md border border-border p-0.5 text-xs" role="radiogroup" aria-label="Row density">
	{#each LEVELS as level (level.value)}
		<button
			type="button"
			role="radio"
			aria-checked={preferences.density === level.value}
			class={[
				'rounded-sm px-2 py-1.5',
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
