<script lang="ts">
	// The Execution Controls (docs/ui-plan.md merge 11): the console's one
	// writable-to-a-run surface. It reaches the whole designed workflow by
	// enqueuing — never shelling out, never running a batch in-process (§4
	// "Running the pipeline from the screen"). Dispatch is always the queue,
	// so a store *and* a DSN are required; `--allow-execute` is the switch.
	//
	// Not run-scoped: like the Composer and Resources Console, this screen
	// configures a *new* run rather than reading an existing one. A successful
	// enqueue points the run selector at the run it minted, so the operator can
	// switch straight to the Dashboard or Corpus Inspector to watch it drain.
	import {
		getExecutionStatus,
		enqueueExtraction,
		EnqueueRefused,
		type ExecutionStatus,
		type EnqueueResult
	} from '$lib/api';
	import { runSelection } from '$lib/stores/run.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';

	// `$state<T>()` not a `let: T | null` annotation: TS narrows the latter to
	// `null`, so the `blocker` $derived below reads it as `never` (the same
	// svelte-check trap the Chunk Inspector hit).
	let status = $state<ExecutionStatus | null>(null);
	let loading = $state(true);
	let loadError: string | null = $state(null);

	// The configure-and-run form. Defaults mirror `EnqueueRequest`'s server-side
	// ones (`batch_size=50`, `max_attempts=3`); an empty `run_id` mints a fresh
	// timestamped id server-side rather than sending a blank string.
	let inputPrefix = $state('');
	let runId = $state('');
	let batchSize = $state(50);
	let maxAttempts = $state(3);

	let enqueuing = $state(false);
	let result = $state<EnqueueResult | null>(null);
	let enqueueError: string | null = $state(null);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// Which of the three requirements is missing, named so the operator sees
	// the actionable fix rather than a bare "disabled". Order matches the
	// guard's (`ui/execute.py`): the switch first (a deliberate choice), then
	// the store and queue (wiring gaps).
	let blocker = $derived.by((): { label: string; detail: string } | null => {
		if (!status || status.can_execute) return null;
		if (!status.allow_execute) {
			return {
				label: 'Audit-only',
				detail:
					'This console was started without --allow-execute, so it can configure and ' +
					'audit but not dispatch. Restart it with --allow-execute to enable this screen.'
			};
		}
		if (!status.has_store) {
			return {
				label: 'No store',
				detail:
					'Execution dispatches through a shared object store; this console reads a local ' +
					'output_root. Point it at a --store (or $WOMBLEX_STORE_URI) to enqueue work.'
			};
		}
		return {
			label: 'No queue',
			detail:
				'Execution dispatches through the job queue; no DSN is configured. Set one ' +
				'(--dsn / $WOMBLEX_DB_DSN) to enqueue work.'
		};
	});

	$effect(() => {
		loading = true;
		loadError = null;
		getExecutionStatus()
			.then((body) => (status = body))
			.catch((err) => (loadError = message(err)))
			.finally(() => (loading = false));
	});

	async function submit(event: SubmitEvent): Promise<void> {
		event.preventDefault();
		if (enqueuing || !inputPrefix.trim()) return;
		enqueuing = true;
		result = null;
		enqueueError = null;
		try {
			result = await enqueueExtraction({
				input_prefix: inputPrefix.trim(),
				run_id: runId.trim() || undefined,
				batch_size: batchSize,
				max_attempts: maxAttempts
			});
			// Point the rest of the console at the run just planned, and refresh
			// the run list so the selector shows it. `load()` reconciles the
			// stored selection, so `select()` first makes the new id survive it.
			runSelection.select(result.run_id);
			await runSelection.load();
			runSelection.select(result.run_id);
		} catch (err) {
			// A capability change since load (e.g. the switch was flipped off)
			// surfaces here as 403/409; refresh the status so the form disables
			// and the blocker banner explains why, matching what the server saw.
			if (err instanceof EnqueueRefused && err.status !== 400) {
				getExecutionStatus()
					.then((body) => (status = body))
					.catch(() => {});
			}
			enqueueError = message(err);
		} finally {
			enqueuing = false;
		}
	}
</script>

<div class="flex h-full flex-col gap-4 p-6">
	<div class="flex items-center justify-between gap-2">
		<h1 class="font-display text-2xl">Execution Controls</h1>
		{#if status}
			<StatusPill
				status={status.can_execute ? 'done' : 'warning'}
				label={status.can_execute ? 'Ready to dispatch' : (blocker?.label ?? 'Disabled')}
			/>
		{/if}
	</div>

	{#if loading}
		<p class="text-sm text-muted-foreground">Loading execution status…</p>
	{:else if loadError}
		<p class="text-sm text-status-failed">{loadError}</p>
	{:else if status}
		{#if blocker}
			<!-- Dispatch is off or unwired. Name the fix; the form below stays
			     visible but disabled so the operator sees what it would do. -->
			<div class="rounded-md border border-status-warning/40 bg-status-warning/10 p-4">
				<p class="font-display text-sm">{blocker.label}</p>
				<p class="mt-1 max-w-prose text-xs text-muted-foreground">{blocker.detail}</p>
			</div>
		{/if}

		<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
			<div>
				<h2 class="font-display text-sm">Configure and run</h2>
				<p class="mt-1 max-w-prose text-xs text-muted-foreground">
					Plans an extraction run into the job queue — lists supported documents under the
					prefix, splits them into batches, and enqueues one idempotent row each. Workers the
					platform brings up do the work; nothing runs in this process.
				</p>
			</div>

			<form class="flex flex-col gap-3" onsubmit={submit}>
				<label class="flex flex-col gap-1 text-xs">
					<span class="text-muted-foreground">Input prefix</span>
					<input
						type="text"
						bind:value={inputPrefix}
						placeholder="inbox"
						disabled={!status.can_execute || enqueuing}
						class="rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs
							focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
							disabled:opacity-50"
					/>
					<span class="text-muted-foreground">
						Store-relative — the enqueue lists <code class="font-mono">&lt;store&gt;/&lt;prefix&gt;</code>.
					</span>
				</label>

				<label class="flex flex-col gap-1 text-xs">
					<span class="text-muted-foreground">Run id <span class="opacity-60">(optional)</span></span>
					<input
						type="text"
						bind:value={runId}
						placeholder="mint a fresh timestamped id"
						disabled={!status.can_execute || enqueuing}
						class="rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs
							focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
							disabled:opacity-50"
					/>
					<span class="text-muted-foreground">
						Supplying an existing run id resumes it — enqueue inserts only its missing batches.
					</span>
				</label>

				<div class="grid grid-cols-2 gap-3">
					<label class="flex flex-col gap-1 text-xs">
						<span class="text-muted-foreground">Batch size</span>
						<input
							type="number"
							min="1"
							bind:value={batchSize}
							disabled={!status.can_execute || enqueuing}
							class="rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs
								focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
								disabled:opacity-50"
						/>
					</label>
					<label class="flex flex-col gap-1 text-xs">
						<span class="text-muted-foreground">Max attempts</span>
						<input
							type="number"
							min="1"
							bind:value={maxAttempts}
							disabled={!status.can_execute || enqueuing}
							class="rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs
								focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
								disabled:opacity-50"
						/>
					</label>
				</div>

				<div>
					<button
						type="submit"
						disabled={!status.can_execute || enqueuing || !inputPrefix.trim()}
						class="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground
							hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2
							focus-visible:ring-ring disabled:opacity-50 disabled:hover:bg-primary"
					>
						{enqueuing ? 'Enqueuing…' : 'Enqueue run'}
					</button>
				</div>
			</form>

			{#if enqueueError}
				<p class="text-xs text-status-failed">{enqueueError}</p>
			{/if}

			{#if result}
				<!-- The queue is now the source of truth; this is the receipt.
				     `newly_enqueued < batch_count` means a resume that skipped
				     already-planned batches. The run selector is pointed here,
				     so the Dashboard and Corpus Inspector track it from now. -->
				<dl
					class="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-status-done/40
						bg-status-done/10 p-3 text-xs sm:grid-cols-4"
				>
					<div class="col-span-2 sm:col-span-4">
						<dt class="text-muted-foreground">Run id</dt>
						<dd class="font-mono">{result.run_id}</dd>
					</div>
					<div>
						<dt class="text-muted-foreground">Documents</dt>
						<dd class="font-mono">{result.document_count}</dd>
					</div>
					<div>
						<dt class="text-muted-foreground">Batches</dt>
						<dd class="font-mono">{result.batch_count}</dd>
					</div>
					<div>
						<dt class="text-muted-foreground">Newly enqueued</dt>
						<dd class="font-mono">{result.newly_enqueued}</dd>
					</div>
					<div class="col-span-2 sm:col-span-4">
						<dt class="text-muted-foreground">Shard prefix</dt>
						<dd class="truncate font-mono" title={result.shard_prefix}>{result.shard_prefix}</dd>
					</div>
				</dl>
				<p class="text-xs text-muted-foreground">
					Selected this run — open the <a href="/dashboard" class="underline">Dashboard</a> to watch
					the queue drain, or the <a href="/corpus" class="underline">Corpus Inspector</a> as shards
					land.
				</p>
			{/if}
		</section>

		<!-- "Log streaming" is the queue's own job-status transitions plus the
		     per-stage checkpoints the Dashboard already reads (§4 "Live local-run
		     progress") — a batch-granular feed, not a fabricated line-by-line log.
		     So this screen dispatches; the Dashboard is where a run is watched. -->
		<p class="max-w-prose text-xs text-muted-foreground">
			Progress is batch-granular and lives on the
			<a href="/dashboard" class="underline">Dashboard</a> — the queue's job-status transitions and
			the per-stage checkpoints, not a line-by-line log the pipeline does not emit.
		</p>
	{/if}
</div>
