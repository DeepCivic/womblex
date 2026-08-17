// Thin fetch wrapper over the read API `womblex.ui.routes` serves
// (docs/ui-plan.md §3). No client-side caching or state management beyond
// what each screen needs — the console reads shard artefacts, it doesn't
// mutate them, so there is nothing to keep in sync.

export interface RunSummary {
	run_id: string;
	document_count: number;
	stages: string[];
	created_at: string | null;
	updated_at: string | null;
}

export async function listRuns(fetchImpl: typeof fetch = fetch): Promise<RunSummary[]> {
	const resp = await fetchImpl('/api/runs');
	if (!resp.ok) throw new Error(`GET /api/runs: ${resp.status}`);
	const body = (await resp.json()) as { runs: RunSummary[] };
	return body.runs;
}

// One row of MANIFEST_SCHEMA — the Corpus Inspector's documents grid.
export interface ManifestDocument {
	source_hash: string;
	collection_id: string;
	doc_id: string;
	filename: string;
	ext: string;
	extraction_method: string;
	elements_count: number;
	table_cells_count: number;
	form_fields_count: number;
	status: string;
	error: string;
	extracted_at_iso: string;
	parser_version: string;
}

export async function getManifest(
	runId: string,
	fetchImpl: typeof fetch = fetch
): Promise<ManifestDocument[]> {
	const resp = await fetchImpl(`/api/runs/${encodeURIComponent(runId)}/manifest`);
	if (!resp.ok) throw new Error(`GET /api/runs/${runId}/manifest: ${resp.status}`);
	const body = (await resp.json()) as { documents: ManifestDocument[] };
	return body.documents;
}

// Which documents have a sidecar row for one stage (docs/ui-plan.md §3
// "lifecycle checkpoints are sidecar presence").
export async function getStagePresence(
	runId: string,
	stage: string,
	fetchImpl: typeof fetch = fetch
): Promise<string[]> {
	const resp = await fetchImpl(
		`/api/runs/${encodeURIComponent(runId)}/stage-presence/${encodeURIComponent(stage)}`
	);
	if (!resp.ok) throw new Error(`GET .../stage-presence/${stage}: ${resp.status}`);
	const body = (await resp.json()) as { source_hashes: string[] };
	return body.source_hashes;
}

// The Chunk Inspector's payload for one document (docs/ui-plan.md merge 6):
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
}

// Entity mention. The reader re-keys the sharded layout's `document_id` onto
// `source_hash` before serialising, so it joins like every other overlay.
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

// One detected PII span, located within a chunk: slice `chunk.text[start:end]`.
export interface PiiSpan {
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
export interface MoneySpan {
	source_hash: string;
	locus: string;
	text_source: string;
	start_char: number;
	end_char: number;
	page: number | null;
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
	if (!resp.ok) throw new Error(`GET /api/runs/${runId}/chunks/${sourceHash}: ${resp.status}`);
	return (await resp.json()) as ChunkDetail;
}

// `ShardAuditReport.as_dict()` — the verify-shards action's result.
export interface ShardAudit {
	shard_dir: string;
	source_count: number | null;
	manifest_row_count: number;
	shard_count: number;
	status_error_rows: number;
	zero_elem_docs: number;
	empty_hashes: number;
	dupe_hashes: number;
	total_elements: number;
	methods: Record<string, number>;
	kind_counts: Record<string, number>;
	corrupted_batches: string[];
}

export async function getAudit(
	runId: string,
	fetchImpl: typeof fetch = fetch
): Promise<ShardAudit> {
	const resp = await fetchImpl(`/api/runs/${encodeURIComponent(runId)}/audit`);
	if (!resp.ok) throw new Error(`GET /api/runs/${runId}/audit: ${resp.status}`);
	return (await resp.json()) as ShardAudit;
}

// The Resources Console's connection cards (docs/ui-plan.md merge 10). Cheap,
// network-free reads — each card's live check is a separate `test*` call.
export interface StoreOptions {
	credentials_configured: boolean;
	endpoint_url: string | null;
	region: string | null;
}

export interface StoreCard {
	kind: 'local' | 'remote';
	uri: string;
	is_object_store: boolean;
	// Empty in local mode — there are no fsspec storage options for a plain
	// directory. Partial rather than required, so reading a field outside the
	// `kind === 'remote'` guard is a type error rather than a runtime undefined.
	options: Partial<StoreOptions>;
}

export interface QueueCard {
	configured: boolean;
	dsn_masked: string | null;
}

export interface IsaacusEndpoint {
	name: string;
	region: string | null;
	models: string[] | null;
}

export interface IsaacusCard {
	deployment: 'hosted' | 'sagemaker';
	endpoints: IsaacusEndpoint[];
	api_key_configured: boolean;
	api_key_masked: string | null;
	models_checked: string[];
	unserved_models: string[];
}

export interface ResourcesCards {
	store: StoreCard;
	queue: QueueCard;
	isaacus: IsaacusCard;
}

export async function getResources(fetchImpl: typeof fetch = fetch): Promise<ResourcesCards> {
	const resp = await fetchImpl('/api/resources');
	if (!resp.ok) throw new Error(`GET /api/resources: ${resp.status}`);
	return (await resp.json()) as ResourcesCards;
}

export interface ReachabilityResult {
	reachable: boolean;
	error: string | null;
}

export async function testStoreConnection(
	fetchImpl: typeof fetch = fetch
): Promise<ReachabilityResult> {
	const resp = await fetchImpl('/api/resources/test/store', { method: 'POST' });
	if (!resp.ok) throw new Error(`POST /api/resources/test/store: ${resp.status}`);
	return (await resp.json()) as ReachabilityResult;
}

// Fleet + queue-depth state, from the same `JobQueue` views the Dashboard
// reads (docs/ui-plan.md merge 8) — the queue card's "test" action doubles as
// that read. `WorkerState` / `Throughput` field names mirror `cloud/queue.py`
// exactly (`asdict()` is what the API serialises).
export interface QueueWorker {
	worker_id: string;
	running: number;
	oldest_locked_at: string | null;
	newest_locked_at: string | null;
}

export interface QueueThroughput {
	window_seconds: number;
	completed: number;
	per_minute: number;
	last_completed_at: string | null;
}

export interface QueueTestResult extends ReachabilityResult {
	queue: {
		stats: Record<string, number>;
		total: number;
		workers: QueueWorker[];
		throughput: QueueThroughput;
	} | null;
}

export async function testQueueConnection(
	fetchImpl: typeof fetch = fetch
): Promise<QueueTestResult> {
	const resp = await fetchImpl('/api/resources/test/queue', { method: 'POST' });
	if (!resp.ok) throw new Error(`POST /api/resources/test/queue: ${resp.status}`);
	return (await resp.json()) as QueueTestResult;
}

// The Pipeline Composer (docs/ui-plan.md merge 9). Neither read is run-scoped:
// the DAG comes from `STAGE_CONTRACTS` and the form's fields from
// `WomblexConfig`'s JSON Schema, so the frontend hand-codes neither the stage
// ordering nor the config field list — §3's "do not hand-code the DAG".
export interface ConditionalInput {
	suffix: string;
	reason: string;
	strict: boolean;
}

export interface StageNode {
	id: string;
	scope: string | null;
	mutation: string | null;
	needs_isaacus_api: boolean;
	checkpoint_dirname: string | null;
	required_inputs: string[];
	conditional_inputs: ConditionalInput[];
	outputs: string[];
}

export interface StageEdge {
	from: string;
	to: string;
	suffixes: string[];
}

export interface StageGraph {
	nodes: StageNode[];
	edges: StageEdge[];
}

export async function getStageGraph(fetchImpl: typeof fetch = fetch): Promise<StageGraph> {
	const resp = await fetchImpl('/api/composer/graph');
	if (!resp.ok) throw new Error(`GET /api/composer/graph: ${resp.status}`);
	return (await resp.json()) as StageGraph;
}
