<script lang="ts">
	import { runSelection } from '$lib/stores/run.svelte';

	function onChange(event: Event): void {
		runSelection.select((event.target as HTMLSelectElement).value || null);
	}
</script>

<label class="flex items-center gap-2 text-sm">
	<span class="sr-only">Selected run</span>
	{#if runSelection.loading}
		<span class="text-muted-foreground">Loading runs…</span>
	{:else if runSelection.error}
		<span class="text-status-failed">{runSelection.error}</span>
	{:else if runSelection.runs.length === 0}
		<span class="text-muted-foreground">No runs found</span>
	{:else}
		<select
			class="rounded-md border border-border bg-surface-raised px-2 py-1 font-mono text-xs
				focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
			value={runSelection.selectedRunId ?? ''}
			onchange={onChange}
		>
			{#each runSelection.runs as run (run.run_id)}
				<option value={run.run_id}>{run.run_id} ({run.document_count} docs)</option>
			{/each}
		</select>
	{/if}
</label>
