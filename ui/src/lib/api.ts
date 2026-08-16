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
