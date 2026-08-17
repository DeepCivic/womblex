<script lang="ts">
	// The Chunk Inspector (docs/ui-plan.md merge 6). Pick a document from the
	// run's manifest, then read its chunks and overlays from
	// `/api/runs/{id}/chunks/{source_hash}` and render one `ChunkCard` each.
	//
	// The endpoint is keyed on a single `source_hash`, not the whole run —
	// chunk / entity / PII / money sidecars span the entire corpus, so a
	// per-document read is what keeps this cheap (the reader pushes the
	// predicate into parquet; see readers.py `_remote_chunk_detail`). The
	// document list therefore comes from the manifest, exactly as the Corpus
	// Inspector's grid does.
	import { runSelection } from '$lib/stores/run.svelte';
	import {
		getManifest,
		getChunkDetail,
		type ManifestDocument,
		type ChunkDetail,
		type MoneySpan,
		type ChunkQuality
	} from '$lib/api';
	import ChunkCard from '$lib/components/ChunkCard.svelte';

	let documents: ManifestDocument[] = $state([]);
	let documentsError: string | null = $state(null);
	let selectedHash: string | null = $state(null);

	let detail: ChunkDetail | null = $state(null);
	let loading = $state(false);
	let error: string | null = $state(null);

	let show = $state({ pii: true, entities: true, money: true });

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	let selectedDoc = $derived(documents.find((d) => d.source_hash === selectedHash) ?? null);

	// Group the flat overlay lists by chunk so each card gets only its own.
	// PII carries `chunk_index` directly; entities the same (-1 when the
	// mention didn't map to a chunk, so those are simply never claimed).
	let piiByChunk = $derived(groupBy(detail?.pii_spans ?? [], (s) => s.chunk_index));
	let entitiesByChunk = $derived(groupBy(detail?.entities ?? [], (e) => e.chunk_index));
	// Quality is one row per chunk (or none, if the stage hasn't run).
	let qualityByChunk = $derived(
		new Map<number, ChunkQuality>((detail?.quality ?? []).map((q) => [q.chunk_index, q]))
	);

	function groupBy<T>(rows: T[], key: (row: T) => number): Map<number, T[]> {
		const map = new Map<number, T[]>();
		for (const row of rows) {
			const k = key(row);
			const bucket = map.get(k);
			if (bucket) bucket.push(row);
			else map.set(k, [row]);
		}
		return map;
	}

	// Money spans anchor to narrative character offsets, not a `chunk_index`.
	// A span belongs to the chunk whose [start_char, end_char) range contains
	// its start — the same overlap test `graph_refresh` uses to map mentions to
	// chunks. Narrative-only by the time it reaches us (the reader drops cell
	// loci), so every span has a `start_char` to place.
	function moneyForChunk(chunk: { start_char: number; end_char: number }): MoneySpan[] {
		return (detail?.money_spans ?? []).filter(
			(m) => m.start_char >= chunk.start_char && m.start_char < chunk.end_char
		);
	}

	let counts = $derived({
		chunks: detail?.chunks.length ?? 0,
		entities: detail?.entities.length ?? 0,
		pii: detail?.pii_spans.length ?? 0,
		money: detail?.money_spans.length ?? 0
	});

	// The run's document list. Same manifest the Corpus Inspector reads; a
	// superseded response is dropped by the teardown flag, as there.
	$effect(() => {
		const runId = runSelection.selectedRunId;
		selectedHash = null;
		detail = null;
		if (!runId) {
			documents = [];
			return;
		}
		let cancelled = false;
		documentsError = null;
		getManifest(runId)
			.then((docs) => {
				if (cancelled) return;
				documents = docs;
				selectedHash = docs[0]?.source_hash ?? null;
			})
			.catch((err) => {
				if (!cancelled) documentsError = message(err);
			});
		return () => {
			cancelled = true;
		};
	});

	// The selected document's chunks + overlays.
	$effect(() => {
		const runId = runSelection.selectedRunId;
		const hash = selectedHash;
		if (!runId || !hash) {
			detail = null;
			return;
		}
		let cancelled = false;
		loading = true;
		error = null;
		getChunkDetail(runId, hash)
			.then((body) => {
				if (!cancelled) detail = body;
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
</script>

<div class="flex h-full flex-col gap-4 p-6">
	<h1 class="font-display text-2xl">Chunk Inspector</h1>

	{#if !runSelection.selectedRunId}
		<p class="text-sm text-muted-foreground">Select a run to inspect its chunks.</p>
	{:else if documentsError}
		<p class="text-sm text-status-failed">{documentsError}</p>
	{:else if documents.length === 0}
		<p class="text-sm text-muted-foreground">This run has no documents.</p>
	{:else}
		<div class="flex flex-wrap items-center gap-4">
			<label class="flex items-center gap-2 text-sm">
				<span class="text-muted-foreground">Document</span>
				<select
					bind:value={selectedHash}
					class="min-w-0 max-w-md truncate rounded-md border border-border bg-background px-2 py-1
						text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
				>
					{#each documents as doc (doc.source_hash)}
						<option value={doc.source_hash}>{doc.filename || doc.doc_id}</option>
					{/each}
				</select>
			</label>

			<!-- Overlay toggles. Each is off-limits when the run has none of that
				 kind — a switch that can only turn nothing on is noise. -->
			<div class="flex items-center gap-3 text-xs">
				<label class="flex items-center gap-1.5" class:opacity-50={counts.pii === 0}>
					<input type="checkbox" bind:checked={show.pii} disabled={counts.pii === 0} class="accent-accent" />
					PII ({counts.pii})
				</label>
				<label class="flex items-center gap-1.5" class:opacity-50={counts.entities === 0}>
					<input
						type="checkbox"
						bind:checked={show.entities}
						disabled={counts.entities === 0}
						class="accent-accent"
					/>
					Entities ({counts.entities})
				</label>
				<label class="flex items-center gap-1.5" class:opacity-50={counts.money === 0}>
					<input type="checkbox" bind:checked={show.money} disabled={counts.money === 0} class="accent-accent" />
					Money ({counts.money})
				</label>
			</div>
		</div>

		{#if selectedDoc}
			<p class="font-mono text-xs text-muted-foreground">
				{selectedDoc.source_hash} · {counts.chunks} chunks
			</p>
		{/if}

		<div class="min-h-0 flex-1 overflow-auto">
			{#if loading}
				<p class="text-sm text-muted-foreground">Loading chunks…</p>
			{:else if error}
				<p class="text-sm text-status-failed">{error}</p>
			{:else if detail && detail.chunks.length === 0}
				<!-- A present-but-empty answer: the chunk stage hasn't run for this
					 document, or it produced none. Distinct from an error. -->
				<div class="rounded-md border border-border bg-surface-sunken p-8 text-center text-sm text-muted-foreground">
					No chunks for this document. The chunk stage may not have run yet.
				</div>
			{:else if detail}
				<div class="flex flex-col gap-3">
					{#each detail.chunks as chunk (chunk.chunk_index)}
						<ChunkCard
							{chunk}
							pii={piiByChunk.get(chunk.chunk_index) ?? []}
							entities={entitiesByChunk.get(chunk.chunk_index) ?? []}
							money={moneyForChunk(chunk)}
							quality={qualityByChunk.get(chunk.chunk_index) ?? null}
							{show}
						/>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>
