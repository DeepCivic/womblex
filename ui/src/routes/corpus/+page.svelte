<script lang="ts">
	import { runSelection } from '$lib/stores/run.svelte';
	import { getManifest, getStagePresence, type ManifestDocument } from '$lib/api';
	import DocumentGrid from '$lib/components/DocumentGrid.svelte';
	import CheckpointSwitcher from '$lib/components/CheckpointSwitcher.svelte';
	import AuditPanel from '$lib/components/AuditPanel.svelte';

	let documents: ManifestDocument[] = $state([]);
	let loading = $state(false);
	let error: string | null = $state(null);
	let checkpointStage: string | null = $state(null);
	let checkpointPresent: Set<string> | null = $state(null);
	let checkpointError: string | null = $state(null);
	let failedOnly = $state(false);

	let selectedRun = $derived(
		runSelection.runs.find((r) => r.run_id === runSelection.selectedRunId) ?? null
	);
	let failedCount = $derived(documents.filter((d) => d.status !== 'completed').length);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// Both effects fire again when the run selector changes, so a slow response
	// for the run the user just left can land after the new run's. `cancelled`
	// is set by the effect's own teardown, which Svelte runs before the re-run
	// — so a superseded response is dropped rather than overwriting the grid.
	$effect(() => {
		const runId = runSelection.selectedRunId;
		checkpointStage = null;
		if (!runId) {
			documents = [];
			return;
		}
		let cancelled = false;
		loading = true;
		error = null;
		getManifest(runId)
			.then((docs) => {
				if (!cancelled) documents = docs;
			})
			.catch((err) => {
				if (!cancelled) error = message(err);
			})
			.finally(() => {
				if (!cancelled) loading = false;
			});
		return () => {
			cancelled = true;
		};
	});

	// Which documents have a sidecar at the selected checkpoint stage. Its
	// failure is kept separate from `error`: presence is an annotation on the
	// grid, so losing it must not take the documents down with it.
	$effect(() => {
		const runId = runSelection.selectedRunId;
		const stage = checkpointStage;
		checkpointError = null;
		if (!runId || !stage) {
			checkpointPresent = null;
			return;
		}
		let cancelled = false;
		getStagePresence(runId, stage)
			.then((hashes) => {
				if (!cancelled) checkpointPresent = new Set(hashes);
			})
			.catch((err) => {
				if (cancelled) return;
				checkpointPresent = null;
				checkpointError = message(err);
			});
		return () => {
			cancelled = true;
		};
	});
</script>

<div class="flex h-full flex-col gap-4 p-6">
	<div class="flex items-start justify-between gap-4">
		<h1 class="font-display text-2xl">Corpus Inspector</h1>
		{#if runSelection.selectedRunId}
			<AuditPanel runId={runSelection.selectedRunId} />
		{/if}
	</div>

	{#if !runSelection.selectedRunId}
		<p class="text-sm text-muted-foreground">Select a run to inspect its documents.</p>
	{:else if loading}
		<p class="text-sm text-muted-foreground">Loading documents…</p>
	{:else if error}
		<p class="text-sm text-status-failed">{error}</p>
	{:else}
		<div class="flex flex-wrap items-center justify-between gap-3">
			<CheckpointSwitcher stages={selectedRun?.stages ?? []} bind:selected={checkpointStage} />
			<label class="flex items-center gap-2 text-xs text-muted-foreground">
				<input type="checkbox" bind:checked={failedOnly} class="accent-accent" />
				Failed only ({failedCount})
			</label>
		</div>
		{#if checkpointError}
			<p class="text-xs text-status-failed">{checkpointError}</p>
		{/if}
		<div class="min-h-0 flex-1 overflow-auto rounded-md border border-border bg-surface-raised">
			<DocumentGrid {documents} {checkpointStage} {checkpointPresent} {failedOnly} />
		</div>
	{/if}
</div>
