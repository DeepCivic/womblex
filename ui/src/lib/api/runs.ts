import { errorDetail } from './http';

export interface RunSummary {
	run_id: string;
	document_count: number;
	stages: string[];
	created_at: string | null;
	updated_at: string | null;
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
	if (!resp.ok) throw new Error(await errorDetail(resp, `GET /api/runs/${runId}/manifest`));
	const body = (await resp.json()) as { documents: ManifestDocument[] };
	return body.documents;
}

// Which documents have a sidecar row for one stage — a lifecycle
// checkpoint *is* sidecar presence.
export async function getStagePresence(
	runId: string,
	stage: string,
	fetchImpl: typeof fetch = fetch
): Promise<string[]> {
	const resp = await fetchImpl(
		`/api/runs/${encodeURIComponent(runId)}/stage-presence/${encodeURIComponent(stage)}`
	);
	if (!resp.ok) throw new Error(await errorDetail(resp, `GET .../stage-presence/${stage}`));
	const body = (await resp.json()) as { source_hashes: string[] };
	return body.source_hashes;
}

// One published batch log. `modified` is null
// when the backend cannot report it (some object stores omit it per-listing).
export interface RunLog {
	name: string;
	size: number;
	modified: string | null;
}

// The list of a run's batch logs. A run that exists but has no logs (written
// before workers published them) lists empty rather than 404-ing, so the screen
// tells "no logs yet" apart from "no such run".
export async function listRunLogs(
	runId: string,
	fetchImpl: typeof fetch = fetch
): Promise<RunLog[]> {
	const resp = await fetchImpl(`/api/runs/${encodeURIComponent(runId)}/logs`);
	if (!resp.ok) throw new Error(await errorDetail(resp, `GET /api/runs/${runId}/logs`));
	const body = (await resp.json()) as { logs: RunLog[] };
	return body.logs;
}

/**
 * A requested log the console would not serve. Both a name that fails the
 * `batch-NNNN.log` pattern and a valid-but-absent name land here with a 404
 * carrying `available` — the live list of logs — so the picker re-renders
 * inline with "that log is no longer available" and self-corrects a stale link
 * rather than stranding the operator.
 */
export class RunLogNotFound extends Error {
	available: RunLog[];
	constructor(message: string, available: RunLog[]) {
		super(message);
		this.name = 'RunLogNotFound';
		this.available = available;
	}
}

// The text of one batch log. `download` sets the `?download=1` flag so the same
// endpoint serves both the inline `<pre>` viewer and a file download.
export async function getRunLog(
	runId: string,
	name: string,
	{ download = false, fetchImpl = fetch }: { download?: boolean; fetchImpl?: typeof fetch } = {}
): Promise<string> {
	const query = download ? '?download=1' : '';
	const resp = await fetchImpl(
		`/api/runs/${encodeURIComponent(runId)}/logs/${encodeURIComponent(name)}${query}`
	);
	if (!resp.ok) {
		if (resp.status === 404) {
			const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
			if (detail && typeof detail === 'object' && 'available' in detail) {
				const d = detail as { message?: string; available?: RunLog[] };
				throw new RunLogNotFound(d.message ?? 'log not found', d.available ?? []);
			}
		}
		throw new Error(await errorDetail(resp, `GET /api/runs/${runId}/logs/${name}`));
	}
	return await resp.text();
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
	if (!resp.ok) throw new Error(await errorDetail(resp, `GET /api/runs/${runId}/audit`));
	return (await resp.json()) as ShardAudit;
}
