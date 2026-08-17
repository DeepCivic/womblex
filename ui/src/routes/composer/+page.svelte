<script lang="ts">
	// The Pipeline Composer (docs/ui-plan.md merge 9). Like the Resources
	// Console it is not run-scoped: it shows the pipeline's shape and edits
	// configuration, so it never touches a run's artefacts.
	//
	// This half is the shape. The DAG is served from `STAGE_CONTRACTS`, so
	// "extraction precedes chunking" is drawn from the contracts rather than
	// re-stated in TypeScript (plan §3) — and a stage's declared inputs and
	// outputs are read off the same node, not a second list to keep in sync.
	import { getStageGraph, type StageGraph as StageGraphData } from '$lib/api';
	import StageGraph from '$lib/components/StageGraph.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';

	// `$state<T | null>` rather than an annotation on the `let`: TypeScript
	// narrows the latter to `null` from its initialiser, which makes every
	// top-level `$derived` over it an error on `never`.
	let graph = $state<StageGraphData | null>(null);
	let selected = $state<string | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	let selectedNode = $derived(graph?.nodes.find((n) => n.id === selected) ?? null);

	$effect(() => {
		getStageGraph()
			.then((body) => (graph = body))
			.catch((err) => (error = err instanceof Error ? err.message : String(err)))
			.finally(() => (loading = false));
	});
</script>

<div class="flex h-full flex-col gap-4 overflow-auto p-6">
	<h1 class="font-display text-2xl">Pipeline Composer</h1>

	{#if loading}
		<p class="text-sm text-muted-foreground">Loading pipeline shape…</p>
	{:else if error}
		<p class="text-sm text-status-failed">{error}</p>
	{:else if graph}
		<StageGraph {graph} bind:selected />

		{#if selectedNode}
			<section class="rounded-md border border-border bg-surface-raised p-4 text-xs">
				<div class="mb-2 flex flex-wrap items-center gap-2">
					<h2 class="font-display text-sm">{selectedNode.id}</h2>
					{#if selectedNode.needs_isaacus_api}
						<StatusPill status="warning" label="Isaacus API" />
					{/if}
					<!-- `extract` is neither scoped nor a sidecar producer in the
						 runner's sense: it runs inside `process_batch`, which is why
						 its contract fields come back null. -->
					<span class="text-muted-foreground">
						{selectedNode.scope ?? 'in-batch'} · {selectedNode.mutation ?? 'extraction'}
					</span>
				</div>
				<dl class="grid gap-x-4 gap-y-1 sm:grid-cols-3">
					<dt class="text-muted-foreground">Reads</dt>
					<dt class="text-muted-foreground">Also reads (config-dependent)</dt>
					<dt class="text-muted-foreground">Writes</dt>
					<dd class="font-mono break-all">{selectedNode.required_inputs.join(' ') || '—'}</dd>
					<dd class="font-mono break-all">
						{selectedNode.conditional_inputs.map((c) => c.suffix).join(' ') || '—'}
					</dd>
					<dd class="font-mono break-all">{selectedNode.outputs.join(' ') || '—'}</dd>
				</dl>
			</section>
		{/if}
	{/if}
</div>
