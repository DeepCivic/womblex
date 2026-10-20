import { errorDetail, JSON_POST } from './http';
import type { ConfigObject } from './composer';

// The console's dispatch capability. `ExecutionCapability
// .as_dict()`: whether this deployment can dispatch work, and which of the three
// requirements is missing if not. `can_execute` is the conjunction of
// `has_store && has_ingest && has_queue`; the individual flags
// let the composer name the missing piece. `ingest_uri`/`output_uri` feed the
// composer's read-only deployment-locations strip. There is no stage list: the
// pipeline's shape comes from `getStageGraph()`, which serves it from the
// contracts, and the field this once carried was read by nothing.
export interface ExecutionStatus {
	can_execute: boolean;
	has_store: boolean;
	has_ingest: boolean;
	has_queue: boolean;
	ingest_uri: string | null;
	output_uri: string | null;
}

export async function getExecutionStatus(
	fetchImpl: typeof fetch = fetch
): Promise<ExecutionStatus> {
	const resp = await fetchImpl('/api/execute/status');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/execute/status'));
	return (await resp.json()) as ExecutionStatus;
}

// Reachability + document count of the configured ingest location.
// Serves both the composer's "N documents
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

// The configure-and-run form. `input_prefix` is ingest-relative and optional —
// omitted means the whole configured ingest root, which is the normal case.
// `run_id` omitted mints a fresh timestamped id,
// and supplying an existing one resumes it (enqueue is idempotent on
// `(run_id, batch_num)`).
export interface EnqueueRequest {
	config?: Record<string, unknown>;
	input_prefix?: string | null;
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
 * code so the screen can tell the failure shapes apart without parsing
 * the message: 409 (no store, ingest or queue configured — a wiring gap), 400
 * (bad input, e.g. no documents under the prefix). These mirror
 * `ui/execute.py`'s guard and `ValueError` paths.
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

// The "run downstream stages" press. `config` is the composer's current form
// state: it decides which stages a run wants, and every stage runs under it on
// whichever worker claims it, exactly as `womblex enqueue-stages --config` behaves.
// `run_id` is required: stages run over shards that already exist.
export interface StageDispatchRequest {
	run_id: string;
	config: ConfigObject;
	max_attempts?: number;
}

// `StageDispatchResult.as_dict()`. `stages` is what the config enabled, in
// pipeline order — reported back because the list is derived rather than typed,
// so a press that dispatched nothing (or more than expected) is visible.
// `newly_enqueued` reads 0 on a repeat press: `enqueue_stages` is idempotent
// per `(run_id, stage)`.
export interface StageDispatchResult {
	run_id: string;
	stages: string[];
	newly_enqueued: number;
	shard_prefix: string;
}

export async function dispatchDownstreamStages(
	req: StageDispatchRequest,
	fetchImpl: typeof fetch = fetch
): Promise<StageDispatchResult> {
	const resp = await fetchImpl('/api/execute/stages', {
		...JSON_POST,
		body: JSON.stringify(req)
	});
	if (!resp.ok) {
		// 400 carries Pydantic's error list rather than a string when the posted
		// config would not load; stringify so the screen always has a message.
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		const message =
			typeof detail === 'string'
				? detail
				: detail
					? JSON.stringify(detail)
					: `POST /api/execute/stages: ${resp.status}`;
		throw new EnqueueRefused(resp.status, message);
	}
	return (await resp.json()) as StageDispatchResult;
}
