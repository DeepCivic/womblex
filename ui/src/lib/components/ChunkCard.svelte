<script lang="ts">
	// One chunk (docs/ui-plan.md merge 6): its text with PII spans highlighted
	// inline, plus the entity and money overlays that resolved to this chunk.
	//
	// Only PII spans are chunk-relative — `PII_SPANS_SCHEMA` documents its
	// `start`/`end` as offsets into `chunk.text` (slice `text[start:end]`). So
	// those are the only overlay drawn *on* the text. Entity mentions
	// (`mention_start`/`_end`) and money spans (`start_char`/`_end`) are offsets
	// into the reassembled document narrative, a different coordinate space; the
	// screen groups them onto their chunk (entities by `chunk_index`, money by
	// range containment) and lists them beside the text rather than inventing a
	// mapping the pipeline never wrote (the §1 "surface what exists" rule).
	import type { Chunk, PiiSpan, EntityMention, MoneySpan, ChunkQuality } from '$lib/api';

	let {
		chunk,
		pii,
		entities,
		money,
		quality,
		show
	}: {
		chunk: Chunk;
		pii: PiiSpan[];
		entities: EntityMention[];
		money: MoneySpan[];
		quality: ChunkQuality | null;
		show: { pii: boolean; entities: boolean; money: boolean };
	} = $props();

	interface Segment {
		text: string;
		span: PiiSpan | null;
	}

	// Split `chunk.text` into plain / highlighted segments at the PII span
	// boundaries. Spans are clamped to the text length and sorted; an
	// overlapping or out-of-range span (a drifted sidecar) is dropped rather
	// than corrupting the offsets of the ones after it.
	let segments = $derived.by((): Segment[] => {
		if (!show.pii || pii.length === 0) return [{ text: chunk.text, span: null }];
		const len = chunk.text.length;
		const valid = pii
			.filter((s) => s.start >= 0 && s.end <= len && s.start < s.end)
			.sort((a, b) => a.start - b.start);
		const out: Segment[] = [];
		let cursor = 0;
		for (const span of valid) {
			if (span.start < cursor) continue; // overlaps a span already emitted
			if (span.start > cursor) out.push({ text: chunk.text.slice(cursor, span.start), span: null });
			out.push({ text: chunk.text.slice(span.start, span.end), span });
			cursor = span.end;
		}
		if (cursor < len) out.push({ text: chunk.text.slice(cursor), span: null });
		return out;
	});

	function moneyLabel(m: MoneySpan): string {
		const parts = [m.value ?? m.text];
		if (m.currency) parts.unshift(m.currency);
		if (m.multiplier) parts.push(m.multiplier);
		if (m.modifier) parts.push(`(${m.modifier})`);
		return parts.join(' ');
	}
</script>

<article class="flex flex-col gap-2 rounded-md border border-border bg-surface-raised p-4">
	<header class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
		<span class="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-foreground">
			#{chunk.chunk_index}
		</span>
		<span>{chunk.content_type}</span>
		<span class="tabular-nums">chars {chunk.start_char}–{chunk.end_char}</span>
		{#if chunk.page_start !== null}
			<span class="tabular-nums">
				p{chunk.page_start}{chunk.page_end !== null && chunk.page_end !== chunk.page_start
					? `–${chunk.page_end}`
					: ''}
			</span>
		{/if}
		{#if chunk.has_redaction}
			<span class="rounded-full bg-status-warning px-2 py-0.5 text-status-foreground">redacted</span>
		{/if}
		{#if quality?.is_short}
			<span class="rounded-full bg-status-pending px-2 py-0.5 text-status-foreground">short</span>
		{/if}
		{#if quality?.boilerplate_flag}
			<span class="rounded-full bg-status-pending px-2 py-0.5 text-status-foreground">boilerplate</span>
		{/if}
		{#if quality?.exact_dup_id !== null && quality?.exact_dup_id !== undefined}
			<span class="rounded-full bg-status-warning px-2 py-0.5 text-status-foreground">
				exact dup {quality.exact_dup_id}
			</span>
		{:else if quality?.near_dup_id !== null && quality?.near_dup_id !== undefined}
			<span class="rounded-full bg-status-pending px-2 py-0.5 text-status-foreground">
				near dup {quality.near_dup_id}
			</span>
		{/if}
	</header>

	<!-- The chunk body. `whitespace-pre-wrap` keeps the extraction's own line
		 breaks; a mark carries the PII entity type on hover, not as always-on
		 chrome, so a clean read of the text stays possible. -->
	<p class="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">
		{#each segments as seg, i (i)}
			{#if seg.span}
				<mark
					class="rounded bg-status-warning/40 px-0.5 text-foreground"
					title={`${seg.span.entity_type}${seg.span.replacement ? ` → ${seg.span.replacement}` : ''} (${seg.span.detector})`}
				>
					{seg.text}
				</mark>
			{:else}
				{seg.text}
			{/if}
		{/each}
	</p>

	{#if show.entities && entities.length > 0}
		<div class="flex flex-wrap gap-1.5">
			{#each entities as e (e.entity_id + e.mention_start)}
				<span
					class="inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs"
					title={`${e.entity_label}${e.entity_type ? ` · ${e.entity_type}` : ''}${e.role ? ` · ${e.role}` : ''}`}
				>
					<span class="text-muted-foreground">{e.entity_label}</span>
					<span class="font-medium">{e.name || e.entity_id}</span>
				</span>
			{/each}
		</div>
	{/if}

	{#if show.money && money.length > 0}
		<ul class="flex flex-col gap-1 text-xs">
			{#each money as m (m.start_char)}
				<li class="flex items-baseline gap-2">
					<span class="font-mono font-medium tabular-nums">{moneyLabel(m)}</span>
					<span class="truncate text-muted-foreground" title={m.text}>“{m.text}”</span>
				</li>
			{/each}
		</ul>
	{/if}
</article>
