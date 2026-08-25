<script lang="ts">
	// The Dashboard (docs/ui-plan.md merge 8): queue state and per-stage
	// progress, from two sources the pipeline already writes. Run-scoped like
	// the inspectors — the run selector chooses which run's checkpoints to read
	// and which run to scope the queue counts to. It polls, because a run
	// draining is the thing this screen exists to watch; the poll is paused
	// while the tab is hidden so a backgrounded console stops hitting the queue.
	//
	// Nothing here requeues, cancels or claims: the dashboard *names* a stalled
	// job (§4 "Stalled-job identification"), a worker recovers it. Adding an
	// action would make this a second producer of queue state, which the whole
	// console is built to avoid.
	import {
		getDashboard,
		listRunLogs,
		getRunLog,
		RunLogNotFound,
		type DashboardData,
		type JobRow,
		type RunLog
	} from '$lib/api';
	import { runSelection } from '$lib/stores/run.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';
	import type { Status } from '$lib/status';

	// Poll cadence. Fast enough that a draining queue looks live, slow enough
	// that a routable-but-dead queue (each poll waits out `QUEUE_CONNECT_TIMEOUT`
	// server-side) does not stack requests.
	const POLL_MS = 5000;

	let data = $state<DashboardData | null>(null);
	// Distinguish the first load (show a spinner) from a poll (keep the last
	// good data on screen, surface the error quietly) — a transient blip should
	// not blank a dashboard the operator is reading.
	let firstLoad = $state(true);
	let error: string | null = $state(null);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// The four queue statuses map straight onto the shared pill vocabulary;
	// anything the queue grows later falls back to `pending`'s neutral fill
	// rather than crashing the lookup.
	const STATUS_MAP: Record<string, Status> = {
		pending: 'pending',
		running: 'running',
		done: 'done',
		failed: 'failed'
	};
	function statusOf(name: string): Status {
		return STATUS_MAP[name] ?? 'pending';
	}

	// Ordered so the tiles read left-to-right as a batch's lifecycle, with any
	// unexpected status appended rather than dropped.
	const STATUS_ORDER = ['pending', 'running', 'done', 'failed'];
	let statTiles = $derived.by((): { status: string; count: number }[] => {
		const stats = data?.queue?.stats ?? {};
		const known = STATUS_ORDER.filter((s) => s in stats).map((s) => ({
			status: s,
			count: stats[s]
		}));
		const extra = Object.keys(stats)
			.filter((s) => !STATUS_ORDER.includes(s))
			.map((s) => ({ status: s, count: stats[s] }));
		return [...known, ...extra];
	});

	// Stale rows carry an id set so the job list can flag the same rows inline
	// rather than repeating them — a stale job is by definition a running one,
	// so it is already in the list.
	let staleIds = $derived(new Set((data?.queue?.stale ?? []).map((j) => j.id)));

	function shortId(worker: string): string {
		return worker.length > 24 ? `${worker.slice(0, 21)}…` : worker;
	}

	function ago(iso: string | null): string {
		if (!iso) return '—';
		const then = new Date(iso).getTime();
		if (Number.isNaN(then)) return '—';
		const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
		if (secs < 60) return `${secs}s ago`;
		if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
		if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
		return `${Math.round(secs / 86400)}d ago`;
	}

	function pct(stage: DashboardData['stages'][number]): number | null {
		if (stage.processed <= 0) return null;
		return Math.round((stage.succeeded / stage.processed) * 100);
	}

	// One effect owns both the run scope and the polling timer. It re-runs when
	// the selected run changes; `cancelled` drops a response for the run the
	// user just left (the same superseded-response guard the Corpus Inspector
	// uses), and the interval is cleared on teardown so a run switch doesn't
	// leave two timers polling.
	$effect(() => {
		const runId = runSelection.selectedRunId;
		let cancelled = false;
		firstLoad = true;

		async function poll(): Promise<void> {
			// document may be undefined during SSR; guard so the poll is a no-op
			// there and resumes on the client.
			if (typeof document !== 'undefined' && document.hidden) return;
			try {
				const body = await getDashboard(runId);
				if (cancelled) return;
				data = body;
				error = null;
			} catch (err) {
				if (!cancelled) error = message(err);
			} finally {
				if (!cancelled) firstLoad = false;
			}
		}

		poll();
		const timer = setInterval(poll, POLL_MS);
		return () => {
			cancelled = true;
			clearInterval(timer);
		};
	});

	// --- Logs panel ---------------------------------------------------------
	//
	// Reads once per run selection (logs are static once published, unlike the
	// draining queue), and again when the operator picks a batch. Kept separate
	// from the polling effect so a poll failure never blanks a log the operator
	// is reading, and vice versa. `logsState` distinguishes the three states the
	// plan calls out: no such run's logs yet, an unreadable list, and a selected
	// log that has since gone (self-corrected from the 404's `available` list).
	let logs = $state<RunLog[] | null>(null);
	let logsError: string | null = $state(null);
	let selectedLog: string | null = $state(null);
	let logText: string | null = $state(null);
	let logTextError: string | null = $state(null);
	let logLoading = $state(false);

	$effect(() => {
		const runId = runSelection.selectedRunId;
		let cancelled = false;
		logs = null;
		logsError = null;
		selectedLog = null;
		logText = null;
		logTextError = null;
		if (!runId) return;
		(async () => {
			try {
				const rows = await listRunLogs(runId);
				if (!cancelled) logs = rows;
			} catch (err) {
				if (!cancelled) logsError = message(err);
			}
		})();
		return () => {
			cancelled = true;
		};
	});

	async function openLog(name: string): Promise<void> {
		const runId = runSelection.selectedRunId;
		if (!runId) return;
		selectedLog = name;
		logText = null;
		logTextError = null;
		logLoading = true;
		try {
			logText = await getRunLog(runId, name);
		} catch (err) {
			if (err instanceof RunLogNotFound) {
				// Stale link (a batch requeued under a new number, or logs pruned):
				// re-render the picker from the fresh `available` list and say so.
				logs = err.available;
				selectedLog = null;
				logTextError = 'That log is no longer available.';
			} else {
				logTextError = message(err);
			}
		} finally {
			logLoading = false;
		}
	}

	function downloadUrl(name: string): string {
		const runId = runSelection.selectedRunId ?? '';
		return `/api/runs/${encodeURIComponent(runId)}/logs/${encodeURIComponent(name)}?download=1`;
	}
</script>

{#snippet jobRow(job: JobRow)}
	<tr class="border-t border-border">
		<td class="px-3 py-1.5 font-mono">{job.batch_num}</td>
		<td class="px-3 py-1.5">
			<span class="inline-flex items-center gap-1.5">
				<StatusPill status={statusOf(job.status)} label={job.status} />
				{#if staleIds.has(job.id)}
					<StatusPill status="warning" label="stale" />
				{/if}
			</span>
		</td>
		<td class="px-3 py-1.5 font-mono text-muted-foreground">
			{job.attempts}/{job.max_attempts}
		</td>
		<td class="px-3 py-1.5 font-mono text-muted-foreground" title={job.locked_by ?? ''}>
			{job.locked_by ? shortId(job.locked_by) : '—'}
		</td>
		<td class="px-3 py-1.5 text-muted-foreground" title={job.updated_at ?? ''}>
			{ago(job.updated_at)}
		</td>
		<td class="max-w-xs truncate px-3 py-1.5 text-status-failed" title={job.error ?? ''}>
			{job.error ?? ''}
		</td>
	</tr>
{/snippet}

<div class="flex h-full flex-col gap-4 overflow-auto p-6">
	<div class="flex items-center justify-between gap-2">
		<h1 class="font-display text-2xl">Dashboard</h1>
		{#if data}
			<span class="text-xs text-muted-foreground">
				{runSelection.selectedRunId ? `Run ${runSelection.selectedRunId}` : 'All runs'} · refreshes
				every {POLL_MS / 1000}s
			</span>
		{/if}
	</div>

	{#if firstLoad}
		<p class="text-sm text-muted-foreground">Loading dashboard…</p>
	{:else if error && !data}
		<p class="text-sm text-status-failed">{error}</p>
	{:else if data}
		{#if error}
			<!-- A poll failed but we still have the last good frame. Surface it
			     quietly rather than blanking the screen. -->
			<p class="text-xs text-status-warning">Last refresh failed: {error}</p>
		{/if}

		<!-- Queue half -->
		{#if data.queue}
			<section class="flex flex-col gap-3">
				<h2 class="font-display text-sm">Job queue</h2>
				<!-- KPI tiles: the exact queue counts, plus total and throughput.
				     Throughput is derived from `updated_at` deltas, no new schema
				     (§4 "Real-time throughput"). -->
				<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
					{#each statTiles as tile (tile.status)}
						<div
							class="flex flex-col gap-1 rounded-md border border-border bg-surface-raised p-3"
						>
							<div class="flex items-center gap-1.5">
								<StatusPill status={statusOf(tile.status)} label={tile.status} />
							</div>
							<span class="font-display text-2xl">{tile.count}</span>
						</div>
					{/each}
					<div class="flex flex-col gap-1 rounded-md border border-border bg-surface-raised p-3">
						<span class="text-xs text-muted-foreground">Total</span>
						<span class="font-display text-2xl">{data.queue.total}</span>
					</div>
					<div class="flex flex-col gap-1 rounded-md border border-border bg-surface-raised p-3">
						<span class="text-xs text-muted-foreground">Throughput</span>
						<span class="font-display text-2xl"
							>{data.queue.throughput.per_minute.toFixed(1)}</span
						>
						<span class="text-xs text-muted-foreground">
							/min · {data.queue.throughput.completed} in last
							{Math.round(data.queue.throughput.window_seconds / 60)}m
						</span>
					</div>
				</div>

				<!-- Worker fleet, from `locked_by` on running rows. Not liveness:
				     an exited worker leaves a stale lock, which reads as stalled
				     (§4 "Worker fleet status"). -->
				<div class="flex flex-col gap-2 rounded-md border border-border bg-surface-raised p-4">
					<h3 class="font-display text-xs">Worker fleet</h3>
					{#if data.queue.workers.length > 0}
						<ul class="flex flex-col gap-1 text-xs">
							{#each data.queue.workers as w (w.worker_id)}
								<li class="flex items-center justify-between gap-2">
									<span class="truncate font-mono" title={w.worker_id}>{w.worker_id}</span>
									<span class="shrink-0 text-muted-foreground">
										{w.running} running · oldest {ago(w.oldest_locked_at)}
									</span>
								</li>
							{/each}
						</ul>
					{:else}
						<p class="text-xs text-muted-foreground">No workers currently hold a lock.</p>
					{/if}
				</div>

				<!-- Stale detection: running rows past the threshold, the same
				     ones `requeue_stale` recovers. Named, never acted on. -->
				{#if data.queue.stale.length > 0}
					<div
						class="flex flex-col gap-2 rounded-md border border-status-warning/40 bg-status-warning/10 p-4"
					>
						<h3 class="font-display text-xs">
							{data.queue.stale.length} stalled job{data.queue.stale.length === 1 ? '' : 's'}
						</h3>
						<p class="max-w-prose text-xs text-muted-foreground">
							Locked longer than {Math.round(data.stale_after_seconds / 60)}m — what a worker's
							<code class="font-mono">--stale-timeout</code> requeues. An exited worker leaves a lock
							behind, so this can be orphaned work rather than a busy one. The dashboard names them; a
							worker recovers them.
						</p>
					</div>
				{/if}

				<!-- Job list: `womblex_jobs` itself, newest activity first. Stale
				     rows are flagged inline rather than repeated. -->
				<div class="overflow-auto rounded-md border border-border bg-surface-raised">
					<table class="w-full text-left text-xs">
						<thead class="text-muted-foreground">
							<tr>
								<th class="px-3 py-2 font-medium">Batch</th>
								<th class="px-3 py-2 font-medium">Status</th>
								<th class="px-3 py-2 font-medium">Attempts</th>
								<th class="px-3 py-2 font-medium">Worker</th>
								<th class="px-3 py-2 font-medium">Updated</th>
								<th class="px-3 py-2 font-medium">Error</th>
							</tr>
						</thead>
						<tbody>
							{#each data.queue.jobs as job (job.id)}
								{@render jobRow(job)}
							{/each}
						</tbody>
					</table>
					{#if data.queue.jobs.length === 0}
						<p class="px-3 py-4 text-xs text-muted-foreground">No jobs in the queue.</p>
					{/if}
				</div>
			</section>
		{:else}
			<!-- No queue is a normal local deployment, not a fault (§2): the
			     checkpoint half below still renders. -->
			<div class="rounded-md border border-border bg-surface-raised p-4">
				<h2 class="font-display text-sm">Job queue</h2>
				<p class="mt-1 max-w-prose text-xs text-muted-foreground">
					{#if data.queue_error}
						Queue unreachable: {data.queue_error}. Per-stage checkpoint progress below is unaffected.
					{:else}
						No queue configured — this is a local deployment reading per-stage checkpoints. Set a
						DSN (<code class="font-mono">--dsn</code> /
						<code class="font-mono">$WOMBLEX_DB_DSN</code>) for exact queue counts, fleet and
						throughput.
					{/if}
				</p>
			</div>
		{/if}

		<!-- Checkpoint half: per-stage progress from inside the selected run.
		     Always available, in both deployments — no queue needed. -->
		<section class="flex flex-col gap-3">
			<h2 class="font-display text-sm">Stage progress</h2>
			{#if !runSelection.selectedRunId}
				<p class="text-xs text-muted-foreground">
					Select a run to see its per-stage checkpoint progress.
				</p>
			{:else if data.stages.length === 0}
				<p class="text-xs text-muted-foreground">
					No stage checkpoints yet for this run — no shard stage has written one, or the run's first
					batch has not landed.
				</p>
			{:else}
				<div class="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
					{#each data.stages as stage (stage.stage)}
						<div
							class="flex flex-col gap-2 rounded-md border border-border bg-surface-raised p-4"
						>
							<div class="flex items-center justify-between gap-2">
								<h3 class="font-display text-sm capitalize">{stage.stage}</h3>
								{#if stage.failed > 0}
									<StatusPill status="failed" label={`${stage.failed} failed`} />
								{:else}
									<StatusPill status="done" label="clean" />
								{/if}
							</div>
							<dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
								<dt class="text-muted-foreground">Processed</dt>
								<dd class="text-right font-mono">{stage.processed}</dd>
								<dt class="text-muted-foreground">Succeeded</dt>
								<dd class="text-right font-mono">{stage.succeeded}</dd>
								<dt class="text-muted-foreground">Failed</dt>
								<dd class="text-right font-mono">{stage.failed}</dd>
								<dt class="text-muted-foreground">Last batch</dt>
								<dd class="text-right font-mono">{stage.last_batch}</dd>
								<dt class="text-muted-foreground">Rate</dt>
								<dd class="text-right font-mono">
									{stage.documents_per_minute === null
										? '—'
										: `${stage.documents_per_minute.toFixed(1)}/min`}
								</dd>
								<dt class="text-muted-foreground">Updated</dt>
								<dd class="text-right" title={stage.updated_at}>{ago(stage.updated_at)}</dd>
							</dl>
							{#if pct(stage) !== null}
								<!-- Success fraction of processed. Batch-granular, labelled
								     as such — a smoother bar would be fiction (§4). -->
								<div class="h-1.5 overflow-hidden rounded-full bg-background">
									<div class="h-full bg-status-done" style="width: {pct(stage)}%"></div>
								</div>
							{/if}
						</div>
					{/each}
				</div>
				<p class="max-w-prose text-xs text-muted-foreground">
					Progress is batch-granular: each stage writes a checkpoint once per batch, so this is its
					lifetime average, not an instantaneous rate.
				</p>
			{/if}
		</section>

		<!-- Run logs: the per-document failure
		     lines a worker/`cmd_run` published next to the shards. The `job.error`
		     cell above is left as-is; this is the detail behind it. -->
		<section class="flex flex-col gap-3">
			<h2 class="font-display text-sm">Run logs</h2>
			{#if !runSelection.selectedRunId}
				<p class="text-xs text-muted-foreground">Select a run to read its batch logs.</p>
			{:else if logsError}
				<p class="text-xs text-status-failed">{logsError}</p>
			{:else if logs === null}
				<p class="text-xs text-muted-foreground">Loading logs…</p>
			{:else if logs.length === 0}
				<!-- Run exists but has no logs: it predates this change. Explain rather
				     than showing an unexplained empty panel. -->
				<p class="max-w-prose text-xs text-muted-foreground">
					No batch logs for this run. Logs are published by workers (and
					<code class="font-mono">womblex run</code>) from this version onward — a run created
					before that has none, and its failure reasons are in the
					<span class="font-medium">Error</span> column above.
				</p>
			{:else}
				<div class="flex flex-col gap-3 lg:flex-row">
					<!-- Picker -->
					<ul class="flex shrink-0 flex-col gap-1 lg:w-56">
						{#each logs as log (log.name)}
							<li class="flex items-center justify-between gap-2">
								<button
									type="button"
									class="flex-1 truncate rounded px-2 py-1 text-left font-mono text-xs hover:bg-surface-raised {selectedLog ===
									log.name
										? 'bg-surface-raised font-medium'
										: 'text-muted-foreground'}"
									onclick={() => openLog(log.name)}
								>
									{log.name}
								</button>
								<a
									href={downloadUrl(log.name)}
									class="shrink-0 text-xs text-muted-foreground underline hover:text-foreground"
									title="Download {log.name}">download</a
								>
							</li>
						{/each}
					</ul>

					<!-- Viewer -->
					<div class="min-w-0 flex-1 rounded-md border border-border bg-surface-raised">
						{#if logTextError}
							<p class="px-3 py-2 text-xs text-status-warning">{logTextError}</p>
						{:else if logLoading}
							<p class="px-3 py-2 text-xs text-muted-foreground">Loading {selectedLog}…</p>
						{:else if logText !== null}
							<pre
								class="max-h-96 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs">{logText}</pre>
						{:else}
							<p class="px-3 py-2 text-xs text-muted-foreground">
								Select a batch to read its log. A job that failed before its log was published
								shows its only reason in the <span class="font-medium">Error</span> column above.
							</p>
						{/if}
					</div>
				</div>
			{/if}
		</section>
	{/if}
</div>

