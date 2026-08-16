<script lang="ts">
	// The lifecycle-checkpoint switcher (docs/ui-plan.md merge 5): picks which
	// pipeline stage's sidecar presence annotates the documents grid. `stages`
	// is the run's own `RunSummary.stages` — already in pipeline order — so
	// there is nothing to hand-order here. `null` is the unfiltered "All" option.
	let {
		stages,
		selected = $bindable(null)
	}: { stages: string[]; selected: string | null } = $props();

	let options = $derived<(string | null)[]>([null, ...stages]);
</script>

<div
	class="flex items-center gap-1 overflow-x-auto"
	role="radiogroup"
	aria-label="Lifecycle checkpoint"
>
	{#each options as value (value ?? 'all')}
		<button
			type="button"
			role="radio"
			aria-checked={selected === value}
			class={[
				'shrink-0 rounded-md border px-2.5 py-1 text-xs capitalize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
				selected === value
					? 'border-primary bg-primary text-primary-foreground'
					: 'border-border text-muted-foreground hover:text-foreground'
			]}
			onclick={() => (selected = value)}
		>
			{value ?? 'All'}
		</button>
	{/each}
</div>
