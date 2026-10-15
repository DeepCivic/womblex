import { errorDetail } from './http';

// The Resources Console's connection cards (functional_requirements.md, Requirement 20). Cheap,
// network-free reads — each card's live check is a separate `test*` call.
export interface StoreOptions {
	credentials_configured: boolean;
	// The access-key id masked to its last four characters, for recognition
	// only. Null when no credentials are configured; never the secret.
	credentials_masked: string | null;
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

// The ingest card. Unlike the store, ingest is optional, so `configured` is
// the first thing it reports rather than assuming one of two shapes; `uri` is
// null when unconfigured.
export interface IngestCard {
	configured: boolean;
	uri: string | null;
	is_object_store: boolean;
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
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/resources'));
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
	if (!resp.ok) throw new Error(await errorDetail(resp, 'POST /api/resources/test/store'));
	return (await resp.json()) as ReachabilityResult;
}

export async function testIngest(fetchImpl: typeof fetch = fetch): Promise<ReachabilityResult> {
	const resp = await fetchImpl('/api/resources/test/ingest', { method: 'POST' });
	if (!resp.ok) throw new Error(await errorDetail(resp, 'POST /api/resources/test/ingest'));
	return (await resp.json()) as ReachabilityResult;
}

// Fleet + queue-depth state, from the same `JobQueue` views the Dashboard
// reads (functional_requirements.md, Requirement 16) — the queue card's "test" action doubles as
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
	if (!resp.ok) throw new Error(await errorDetail(resp, 'POST /api/resources/test/queue'));
	return (await resp.json()) as QueueTestResult;
}
