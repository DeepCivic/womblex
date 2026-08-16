<script lang="ts">
	// The Corpus Inspector's verify-shards action (docs/ui-plan.md §3): an
	// on-demand call to the existing `audit_shard_directory` — not a poll,
	// since a full audit reads every shard in the run.
	import { getAudit, type ShardAudit } from '$lib/api';

	let { runId }: { runId: string } = $props();
	let loading = $state(false);
	let error: string | null = $state(null);
	let report: ShardAudit | null = $state(null);

	const METRICS: { label: string; key: keyof ShardAudit }[] = [
		{ label: 'Shards', key: 'shard_count' },
		{ label: 'Manifest rows', key: 'manifest_row_count' },
		{ label: 'Status errors', key: 'status_error_rows' },
		{ label: 'Zero-element docs', key: 'zero_elem_docs' },
		{ label: 'Empty hashes', key: 'empty_hashes' },
		{ label: 'Duplicate hashes', key: 'dupe_hashes' },
		{ label: 'Total elements', key: 'total_elements' }
	];

	async function run(): Promise<void> {
		loading = true;
		error = null;
		try {
			report = await getAudit(runId);
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}
</script>

<div class="flex flex-col items-end gap-2">
	<button
		type="button"
		class="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-foreground/5
			focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
		onclick={run}
		disabled={loading}
	>
		{loading ? 'Verifying…' : 'Verify shards'}
	</button>

	{#if error}
		<p class="text-xs text-status-failed">{error}</p>
	{/if}

	{#if report}
		<!-- `surface-raised`, not `surface-sunken`: the sunken token is lime in
			 light mode and DESIGN.md scopes it to wells (logs, empty states).
			 A stats panel is a panel, and muted labels on lime fail contrast. -->
		<dl
			class="grid w-full grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-border bg-surface-raised p-3 text-xs sm:grid-cols-4"
		>
			{#each METRICS as m (m.key)}
				<div>
					<dt class="text-muted-foreground">{m.label}</dt>
					<dd class="font-mono">{report[m.key]}</dd>
				</div>
			{/each}
			<div>
				<dt class="text-muted-foreground">Corrupted batches</dt>
				<dd class={['font-mono', report.corrupted_batches.length > 0 ? 'text-status-failed' : '']}>
					{report.corrupted_batches.length}
				</dd>
			</div>
		</dl>
		{#if report.corrupted_batches.length > 0}
			<ul class="list-inside list-disc self-stretch text-xs text-status-failed">
				{#each report.corrupted_batches as batchId (batchId)}
					<li class="font-mono">{batchId}</li>
				{/each}
			</ul>
		{/if}
	{/if}
</div>
