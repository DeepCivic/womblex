import { errorDetail } from './http';
import type { QueueThroughput, QueueWorker } from './resources';

// The Dashboard (functional_requirements.md, Requirement 16). Two sources, both already written
// by the pipeline: the job queue (when a DSN is configured) and the per-stage
// checkpoints inside the selected run. `queue` is null whenever there is no
// queue to read — no DSN, or the connection failed — and `queue_error` says
// which; that is a normal local deployment, not a fault, so the `stages` half
// still renders. Field names mirror `cloud/queue.py`'s dataclasses and
// `store/checkpoint.py`'s `CheckpointProgress` (`asdict()` is what the API
// serialises), so a renamed column surfaces as a type error here.

// One workflow row on the run board (`JobRow`): the job list's grain. Timestamps are
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
// `--stale-timeout` recovers: stalled jobs are identified, not acted on.
export interface QueueSection {
	stats: Record<string, number>;
	total: number;
	// The run's status sentence (null without a selected run); the CLI and API print the same one.
	summary: string | null;
	jobs: JobRow[];
	workers: QueueWorker[];
	stale: JobRow[];
	throughput: QueueThroughput;
}

// One stage's checkpoint progress for the selected run, in pipeline order.
// `documents_per_minute` is the run's lifetime average, null when the
// timestamps are too close to divide by — a smoother rate would be fiction.
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
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/dashboard'));
	return (await resp.json()) as DashboardData;
}
