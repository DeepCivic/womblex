<script lang="ts">
	// The Corpus Inspector's table (DESIGN.md "Component conventions" —
	// `DocumentGrid`): real <table> semantics, sticky header, failed rows
	// marked by their StatusPill alone (never a tinted row).
	import StatusPill from './StatusPill.svelte';
	import type { Status } from '$lib/status';
	import type { ManifestDocument } from '$lib/api';

	let {
		documents,
		checkpointStage,
		checkpointPresent,
		failedOnly
	}: {
		documents: ManifestDocument[];
		checkpointStage: string | null;
		checkpointPresent: Set<string> | null;
		failedOnly: boolean;
	} = $props();

	function statusOf(doc: ManifestDocument): { status: Status; label: string } {
		if (doc.status === 'completed') return { status: 'done', label: 'completed' };
		if (doc.status === 'error') return { status: 'failed', label: 'error' };
		return { status: 'pending', label: doc.status || 'unknown' };
	}

	let rows = $derived(documents.filter((d) => !failedOnly || d.status !== 'completed'));
	let columnCount = $derived(checkpointStage ? 8 : 7);
</script>

<!-- No aria-rowcount/aria-rowindex: DESIGN.md ties those to virtualised rows,
	 and every row is in the DOM here, so the browser's own count is correct.
	 They become necessary when virtualisation lands. -->
<table class="w-full border-collapse text-left text-sm">
	<caption class="sr-only">Documents in the selected run</caption>
	<thead class="sticky top-0 z-10 bg-surface-raised">
		<tr class="border-b border-border">
			<th scope="col" class="px-3 py-2 font-medium">Document</th>
			<th scope="col" class="px-3 py-2 font-medium">Ext</th>
			<th scope="col" class="px-3 py-2 font-medium">Method</th>
			<th scope="col" class="px-3 py-2 text-right font-medium">Elements</th>
			<th scope="col" class="px-3 py-2 text-right font-medium">Table cells</th>
			<th scope="col" class="px-3 py-2 text-right font-medium">Form fields</th>
			<th scope="col" class="px-3 py-2 font-medium">Status</th>
			{#if checkpointStage}
				<th scope="col" class="px-3 py-2 font-medium capitalize">{checkpointStage}</th>
			{/if}
		</tr>
	</thead>
	<tbody>
		{#each rows as doc (doc.source_hash)}
			{@const st = statusOf(doc)}
			<!-- Hover tints with `foreground`, not `surface-sunken`: the sunken
				 token is lime in light mode (DESIGN.md role table), which reads as
				 a status colour on a row. -->
			<tr
				class="border-b border-border/60 hover:bg-foreground/5"
				style:height="var(--row-height)"
			>
				<td class="px-3 py-1 font-mono text-xs">
					<div>{doc.filename || doc.doc_id}</div>
					{#if doc.status !== 'completed' && doc.error}
						<div class="text-status-failed">{doc.error}</div>
					{/if}
				</td>
				<td class="px-3 py-1">{doc.ext}</td>
				<td class="px-3 py-1">{doc.extraction_method}</td>
				<td class="px-3 py-1 text-right tabular-nums">{doc.elements_count}</td>
				<td class="px-3 py-1 text-right tabular-nums">{doc.table_cells_count}</td>
				<td class="px-3 py-1 text-right tabular-nums">{doc.form_fields_count}</td>
				<td class="px-3 py-1"><StatusPill status={st.status} label={st.label} /></td>
				{#if checkpointStage}
					<td class="px-3 py-1">
						{#if checkpointPresent?.has(doc.source_hash)}
							<StatusPill status="done" label="present" />
						{:else}
							<span class="text-muted-foreground">&mdash;</span>
						{/if}
					</td>
				{/if}
			</tr>
		{/each}
		{#if rows.length === 0}
			<tr>
				<td colspan={columnCount} class="bg-surface-sunken px-3 py-8 text-center text-muted-foreground">
					No documents match.
				</td>
			</tr>
		{/if}
	</tbody>
</table>
