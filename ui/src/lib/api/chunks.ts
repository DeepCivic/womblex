import { errorDetail } from './http';

// The Chunk Inspector's payload for one document (functional_requirements.md, Requirement 18):
// chunks plus the overlay sidecars, all joined on `source_hash` and
// `chunk_index`. Field names mirror the store schemas the reader serialises
// (`CHUNKS_SCHEMA`, `ENTITY_SCHEMA`, `PII_SPANS_SCHEMA`, `MONEY_SPANS_SCHEMA`,
// `CHUNK_QUALITY_SCHEMA`) so a renamed column surfaces as a type error here.
export interface Chunk {
	source_hash: string;
	chunk_index: number;
	text: string;
	start_char: number;
	end_char: number;
	content_type: string;
	has_redaction: boolean;
	page_start: number | null;
	page_end: number | null;
	elem_order: number | null;
	token_count: number | null;
}

// Entity mention. Joins on `source_hash` like every other overlay.
// `chunk_index` is -1 when the mention did not map to a chunk.
export interface EntityMention {
	source_hash: string;
	entity_id: string;
	entity_label: string;
	name: string;
	entity_type: string;
	role: string;
	mention_start: number;
	mention_end: number;
	chunk_index: number;
}

// The evidence reference every span sidecar carries (store/evidence.py): the
// element a span lies in, its page and box, and character offsets into the text
// named by `text_layer`. All null when the span could not be anchored.
export interface Evidence {
	elem_order: number | null;
	page: number | null;
	bbox: { x: number; y: number; width: number; height: number } | null;
	sheet: string | null;
	cell_row: number | null;
	cell_col: number | null;
	char_start: number | null;
	char_end: number | null;
	text_layer: string | null;
	// How precisely the span is located: span | element | chunk | document.
	anchor_level: 'span' | 'element' | 'chunk' | 'document' | null;
	// Whether the receipt was verified before writing; a `fails` row is written, not refused.
	evidence_check: 'holds' | 'fails' | 'unchecked' | null;
}

// One detected PII span. `start` / `end` are derived by the reader from the
// evidence offsets: the span's position in its chunk, `-1` unless it is located
// to the span (`anchor_level === 'span'`).
export interface PiiSpan extends Evidence {
	source_hash: string;
	chunk_index: number;
	content_type: string;
	start: number;
	end: number;
	text: string;
	entity_type: string;
	entity_id: string;
	detector: string;
	score: number;
	replacement: string;
}

// A monetary amount anchored to narrative text. The reader keeps only
// `locus === 'narrative'` spans (the others anchor to cells with no chunk
// offset) and sends `value` as a string — `decimal128(38,4)` is exact by
// contract and a float would lose that.
export interface MoneySpan extends Evidence {
	source_hash: string;
	locus: string;
	text: string;
	value: string | null;
	currency: string | null;
	modifier: string | null;
	multiplier: string | null;
	negative: boolean;
	confidence: number;
	context: string;
}

// Per-chunk quality annotation (ML-readiness flags + duplicate-cluster ids).
export interface ChunkQuality {
	source_hash: string;
	chunk_index: number;
	content_type: string;
	char_len: number;
	alpha_frac: number;
	is_short: boolean;
	boilerplate_flag: boolean;
	exact_dup_id: number | null;
	near_dup_id: number | null;
}

export interface ChunkDetail {
	run_id: string;
	source_hash: string;
	chunks: Chunk[];
	entities: EntityMention[];
	pii_spans: PiiSpan[];
	money_spans: MoneySpan[];
	quality: ChunkQuality[];
}

export async function getChunkDetail(
	runId: string,
	sourceHash: string,
	fetchImpl: typeof fetch = fetch
): Promise<ChunkDetail> {
	const resp = await fetchImpl(
		`/api/runs/${encodeURIComponent(runId)}/chunks/${encodeURIComponent(sourceHash)}`
	);
	if (!resp.ok) throw new Error(await errorDetail(resp, `GET /api/runs/${runId}/chunks/${sourceHash}`));
	return (await resp.json()) as ChunkDetail;
}
