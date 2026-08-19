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

// Prefer the server's own `detail` (e.g. "run store unreachable: Install s3fs
// to access S3") over a bare status code, so a 503 from a misconfigured store
// reads as a fixable cause in the top banner rather than "GET /api/runs: 503".
async function errorDetail(resp: Response, label: string): Promise<string> {
	const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
	if (typeof detail === 'string' && detail) return detail;
	return `${label}: ${resp.status}`;
}

export async function listRuns(fetchImpl: typeof fetch = fetch): Promise<RunSummary[]> {
	const resp = await fetchImpl('/api/runs');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/runs'));
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

// Where an effective location came from (docs/ui-ingest-plan.md §2): a CLI
// `flag`, an `env` var, or a `saved` operator override. The screen collapses
// `flag`/`env` to one "from environment" chip and shows `saved` as "set here",
// but the three are carried distinctly because that is what a reset restores to.
export type LocationSource = 'flag' | 'env' | 'saved';

export interface StoreCard {
	kind: 'local' | 'remote';
	uri: string;
	is_object_store: boolean;
	// Empty in local mode — there are no fsspec storage options for a plain
	// directory. Partial rather than required, so reading a field outside the
	// `kind === 'remote'` guard is a type error rather than a runtime undefined.
	options: Partial<StoreOptions>;
	source: LocationSource;
	// Whether this console has a writable settings dir — the one thing that
	// makes the location editable at all (docs/ui-ingest-plan.md merge 3a).
	// False in both modes when no `--settings-dir` was configured.
	editable: boolean;
}

// The ingest card. Unlike the store, ingest is optional, so `configured` is
// the first thing it reports rather than assuming one of two shapes; `uri` is
// null when unconfigured (docs/ui-ingest-plan.md merge 2/3a).
export interface IngestCard {
	configured: boolean;
	uri: string | null;
	is_object_store: boolean;
	options: Partial<StoreOptions>;
	source: LocationSource;
	editable: boolean;
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
	// A set env var that looks like `ISAACUS_SAGEMAKER_ENDPOINTS` but is not it
	// (e.g. the singular `…_ENDPOINT`), when the canonical var is unset. The
	// deployment silently falls back to the hosted API, so this turns a bare
	// "No API key" into a named, fixable cause. Null when there is nothing to warn.
	endpoints_typo: string | null;
}

export interface ResourcesCards {
	store: StoreCard;
	ingest: IngestCard;
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

export async function testIngest(fetchImpl: typeof fetch = fetch): Promise<ReachabilityResult> {
	const resp = await fetchImpl('/api/resources/test/ingest', { method: 'POST' });
	if (!resp.ok) throw new Error(`POST /api/resources/test/ingest: ${resp.status}`);
	return (await resp.json()) as ReachabilityResult;
}

// Save / update / clear the operator's ingest and output location override
// (docs/ui-ingest-plan.md merge 3a). A full replace (PUT): each field is a new
// location or `null` ("reset to the flag/env default"), so a caller keeping a
// field must resubmit its current value.
export interface SaveLocationsRequest {
	ingest_uri?: string | null;
	store_uri?: string | null;
}

// The refreshed cards plus the reachability verdicts the save re-ran — so the
// screen re-renders the provenance chips and the "reachable?" state in one
// round trip. Reachability is *reported, not required*: naming a bucket ahead
// of provisioning it is a normal save.
export interface SaveLocationsResult {
	ingest: IngestCard;
	store: StoreCard;
	ingest_test: ReachabilityResult;
	store_test: ReachabilityResult;
}

/**
 * A location edit the console refused. `status` is the HTTP code so the screen
 * tells the failure shapes apart without parsing the message: 409 (no writable
 * settings dir — editing is disabled on this console), 403 (audit-only, a
 * deliberate choice), 400 (a malformed URI, or an ingest/output pair that would
 * overlap). Mirrors `ui/routes/resources.py`'s guard and `ValueError` paths.
 */
export class LocationsRefused extends Error {
	status: number;
	constructor(status: number, detail: string) {
		super(detail);
		this.name = 'LocationsRefused';
		this.status = status;
	}
}

export async function saveLocations(
	req: SaveLocationsRequest,
	fetchImpl: typeof fetch = fetch
): Promise<SaveLocationsResult> {
	const resp = await fetchImpl('/api/resources/locations', {
		method: 'PUT',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(req)
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		throw new LocationsRefused(resp.status, detailToMessage(detail, resp.status));
	}
	return (await resp.json()) as SaveLocationsResult;
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

// The Dashboard (docs/ui-plan.md merge 8). Two sources, both already written
// by the pipeline: the job queue (when a DSN is configured) and the per-stage
// checkpoints inside the selected run. `queue` is null whenever there is no
// queue to read — no DSN, or the connection failed — and `queue_error` says
// which; that is a normal local deployment, not a fault, so the `stages` half
// still renders. Field names mirror `cloud/queue.py`'s dataclasses and
// `store/checkpoint.py`'s `CheckpointProgress` (`asdict()` is what the API
// serialises), so a renamed column surfaces as a type error here.

// One `womblex_jobs` row (`JobRow`): the job list's grain. Timestamps are
// ISO-8601 strings, converted once server-side.
export interface JobRow {
	id: number;
	run_id: string;
	batch_num: number;
	status: string;
	attempts: number;
	max_attempts: number;
	locked_by: string | null;
	locked_at: string | null;
	error: string | null;
	created_at: string | null;
	updated_at: string | null;
}

// The whole queue section, or null when there is no queue to read. `stale` is
// the running rows past `stale_after_seconds` — the same predicate a worker's
// `--stale-timeout` recovers, reported not acted on (§4 "Stalled-job
// identification").
export interface QueueSection {
	stats: Record<string, number>;
	total: number;
	jobs: JobRow[];
	workers: QueueWorker[];
	stale: JobRow[];
	throughput: QueueThroughput;
}

// One stage's checkpoint progress for the selected run, in pipeline order.
// `documents_per_minute` is the run's lifetime average, null when the
// timestamps are too close to divide by — a smoother rate would be fiction
// (§4 "Live local-run progress").
export interface StageProgress {
	stage: string;
	name: string;
	processed: number;
	succeeded: number;
	failed: number;
	last_batch: number;
	started_at: string;
	updated_at: string;
	documents_per_minute: number | null;
}

export interface DashboardData {
	run_id: string | null;
	stale_after_seconds: number;
	queue: QueueSection | null;
	queue_error: string | null;
	stages: StageProgress[];
}

// `run_id` scopes the queue views and selects which run's checkpoints to read;
// omit it for the whole-queue view (then `stages` is empty — checkpoints are
// per-run). A missing run is not a 404: an enqueued run whose first batch has
// not landed has no directory yet, and the empty `stages` is the honest answer.
export async function getDashboard(
	runId: string | null,
	fetchImpl: typeof fetch = fetch
): Promise<DashboardData> {
	const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
	const resp = await fetchImpl(`/api/dashboard${query}`);
	if (!resp.ok) throw new Error(`GET /api/dashboard: ${resp.status}`);
	return (await resp.json()) as DashboardData;
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
	// The `WomblexConfig` section that configures this stage, or null where no
	// single section does (`extract`, `graph-refresh`). Served rather than
	// mapped here so a renamed config section breaks a Python test.
	config_section: string | null;
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

// A named pre-configured pipeline the composer form loads as a starting point
// (e.g. `DEFAULT-Isaacus`: extract → chunk → enrich → build_graph → money).
// `config` is a *partial* `WomblexConfig` — it carries stage toggles/settings
// but never `dataset`/`paths`, which name the run and stay the operator's to
// fill — so applying it merges over the form's current config, leaving those.
// `source` is `builtin` (code, undeletable) or `saved` (operator-authored) —
// the screen offers delete only on the latter.
export interface Preset {
	name: string;
	description: string;
	formats: string[];
	config: ConfigObject;
	source: 'builtin' | 'saved';
}

export async function listPresets(fetchImpl: typeof fetch = fetch): Promise<Preset[]> {
	const resp = await fetchImpl('/api/composer/presets');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/composer/presets'));
	const body = (await resp.json()) as { presets: Preset[] };
	return body.presets;
}

// The save-a-preset form. `config` is the whole composed config; the server
// strips `dataset`/`paths` (a preset is an overlay) and validates the rest
// through the same `WomblexConfig(**raw)` the composer's other endpoints use.
export interface SavePresetRequest {
	name: string;
	description?: string;
	formats?: string[];
	config: ConfigObject;
}

/**
 * Saving a preset the console refused. `status` is the HTTP code so the screen
 * distinguishes the failure shapes without parsing the message: 409 (this
 * console cannot write presets — local mode with no presets dir; remote mode
 * always can), 400 (unsafe name, or a config that would not load as an
 * overlay). Mirrors `ui/routes/composer.py`'s guard and error paths.
 */
export class SavePresetRefused extends Error {
	status: number;
	constructor(status: number, detail: string) {
		super(detail);
		this.name = 'SavePresetRefused';
		this.status = status;
	}
}

export async function savePreset(
	req: SavePresetRequest,
	fetchImpl: typeof fetch = fetch
): Promise<Preset> {
	const resp = await fetchImpl('/api/composer/presets', {
		...JSON_POST,
		body: JSON.stringify(req)
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		throw new SavePresetRefused(resp.status, detailToMessage(detail, resp.status));
	}
	return (await resp.json()) as Preset;
}

export async function deletePreset(
	name: string,
	fetchImpl: typeof fetch = fetch
): Promise<void> {
	const resp = await fetchImpl(`/api/composer/presets/${encodeURIComponent(name)}`, {
		method: 'DELETE'
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		throw new SavePresetRefused(resp.status, detailToMessage(detail, resp.status));
	}
}

// FastAPI's `detail` is a string for our raised HTTPExceptions but Pydantic's
// own list-of-errors for a 400 from a bad overlay — flatten both to one line.
function detailToMessage(detail: unknown, status: number): string {
	if (typeof detail === 'string') return detail;
	if (Array.isArray(detail)) {
		return detail
			.map((e) => {
				const loc = Array.isArray(e?.loc) ? e.loc.join('.') : '';
				return loc ? `${loc}: ${e?.msg ?? ''}` : String(e?.msg ?? e);
			})
			.join('; ');
	}
	return `request failed: ${status}`;
}

// The subset of JSON Schema Pydantic emits for `WomblexConfig`. Typed rather
// than `any` so the form's field resolution is checked; unknown keywords are
// simply not read.
export interface JsonSchema {
	type?: string;
	title?: string;
	description?: string;
	default?: unknown;
	properties?: Record<string, JsonSchema>;
	required?: string[];
	items?: JsonSchema;
	anyOf?: JsonSchema[];
	enum?: unknown[];
	additionalProperties?: boolean | JsonSchema;
	$ref?: string;
	$defs?: Record<string, JsonSchema>;
}

export type ConfigObject = Record<string, unknown>;

export async function getConfigSchema(fetchImpl: typeof fetch = fetch): Promise<JsonSchema> {
	const resp = await fetchImpl('/api/composer/schema');
	if (!resp.ok) throw new Error(`GET /api/composer/schema: ${resp.status}`);
	return (await resp.json()) as JsonSchema;
}

// Pydantic's own error shape (`include_url`/`context`/`input` off server-side).
export interface ConfigError {
	loc: (string | number)[];
	msg: string;
	type: string;
}

export interface ValidationResult {
	valid: boolean;
	errors: ConfigError[];
	// Keys the schema does not claim. Always empty for a config this form
	// built — its fields *are* the schema — so no screen renders them; the
	// field is here because the endpoint reports it.
	unknown_keys: string[];
}

const JSON_POST = { method: 'POST', headers: { 'content-type': 'application/json' } };

export async function validateConfig(
	raw: ConfigObject,
	fetchImpl: typeof fetch = fetch
): Promise<ValidationResult> {
	const resp = await fetchImpl('/api/composer/validate', { ...JSON_POST, body: JSON.stringify(raw) });
	if (!resp.ok) throw new Error(`POST /api/composer/validate: ${resp.status}`);
	return (await resp.json()) as ValidationResult;
}

/** A config `/yaml` refused to render — carries the same errors `/validate` reports. */
export class ConfigInvalid extends Error {
	errors: ConfigError[];
	constructor(errors: ConfigError[]) {
		super('Config is invalid');
		this.name = 'ConfigInvalid';
		this.errors = errors;
	}
}

export async function renderConfigYaml(
	raw: ConfigObject,
	fetchImpl: typeof fetch = fetch
): Promise<string> {
	const resp = await fetchImpl('/api/composer/yaml', { ...JSON_POST, body: JSON.stringify(raw) });
	if (resp.status === 422) {
		throw new ConfigInvalid(((await resp.json()) as { detail: ConfigError[] }).detail);
	}
	if (!resp.ok) throw new Error(`POST /api/composer/yaml: ${resp.status}`);
	return await resp.text();
}

// The Execution Controls (docs/ui-plan.md merge 11) — the one writable-to-a-run
// surface. `ExecutionCapability.as_dict()`: whether this deployment can dispatch
// work, and which of the three requirements is missing if not (§4 "Running the
// pipeline from the screen"). `stages` is `STAGE_NAMES`, served so the frontend
// re-types no stage list.
export interface ExecutionStatus {
	can_execute: boolean;
	audit_only: boolean;
	has_store: boolean;
	has_queue: boolean;
	stages: string[];
}

export async function getExecutionStatus(
	fetchImpl: typeof fetch = fetch
): Promise<ExecutionStatus> {
	const resp = await fetchImpl('/api/execute/status');
	if (!resp.ok) throw new Error(`GET /api/execute/status: ${resp.status}`);
	return (await resp.json()) as ExecutionStatus;
}

// Reachability + document count of the configured ingest location
// (docs/ui-ingest-plan.md merge 2). Serves both the composer's "N documents
// ready" line and the Resources Console's ingest card, using the same recursive
// listing + `SUPPORTED_EXTENSIONS` filter the enqueue does — so the count shown
// is the count that would be enqueued. `uri`/`kind` are null when unconfigured.
export interface IngestPreflight {
	uri: string | null;
	kind: 'remote' | 'local' | null;
	reachable: boolean;
	document_count: number;
	sample: string[];
	error: string | null;
}

export async function getIngestPreflight(
	fetchImpl: typeof fetch = fetch
): Promise<IngestPreflight> {
	const resp = await fetchImpl('/api/execute/ingest');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/execute/ingest'));
	return (await resp.json()) as IngestPreflight;
}

// The configure-and-run form. `input_prefix` is store-relative; `run_id`
// omitted mints a fresh timestamped id, and supplying an existing one resumes
// it (enqueue is idempotent on `(run_id, batch_num)`).
export interface EnqueueRequest {
	input_prefix: string;
	run_id?: string | null;
	batch_size?: number;
	max_attempts?: number;
}

// `EnqueueResult.as_dict()`. `newly_enqueued` distinguishes a fresh run from a
// resume (a resume inserts only the batches a run is missing); `run_id` is what
// the Dashboard and Corpus Inspector are then pointed at to watch it drain.
export interface EnqueueResult {
	run_id: string;
	document_count: number;
	batch_count: number;
	newly_enqueued: number;
	shard_prefix: string;
}

/**
 * An enqueue the console refused or could not serve. `status` carries the HTTP
 * code so the screen can tell the three failure shapes apart without parsing
 * the message: 403 (audit-only, a deliberate choice), 409 (no store or queue
 * configured — a wiring gap), 400 (bad input, e.g. no documents under the
 * prefix). These mirror `ui/execute.py`'s guard and `ValueError` paths.
 */
export class EnqueueRefused extends Error {
	status: number;
	constructor(status: number, detail: string) {
		super(detail);
		this.name = 'EnqueueRefused';
		this.status = status;
	}
}

export async function enqueueExtraction(
	req: EnqueueRequest,
	fetchImpl: typeof fetch = fetch
): Promise<EnqueueResult> {
	const resp = await fetchImpl('/api/execute/enqueue', {
		...JSON_POST,
		body: JSON.stringify(req)
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: string }).detail;
		throw new EnqueueRefused(resp.status, detail ?? `POST /api/execute/enqueue: ${resp.status}`);
	}
	return (await resp.json()) as EnqueueResult;
}
