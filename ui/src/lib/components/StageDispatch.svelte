<script lang="ts">
	// "Run downstream stages" — the console's second write action (issue 5).
	//
	// Deliberately a separate press from the composer's "Enqueue run". A worker
	// will not claim a stage until the run's batches settle, so nothing races;
	// but a *bad* extraction should not spend Isaacus budget on enrich and embed
	// either, so the operator enqueues, watches the Dashboard drain, looks at the
	// run, then presses this. Same split, same reasoning, as `womblex enqueue`
	// versus `womblex enqueue-stages`.
	//
	// Which stages run is neither asked here nor decided here: the server derives
	// it from the config posted with the press, through the same
	// `enabled_downstream_stages` gate the CLI applies. Reading `config` here to
	// pre-draw the list would re-implement that gate (normalise answers to
	// `text_source`, graph-refresh to `chunking_model`) — exactly the drift the
	// composer's "do not hand-code the DAG in the frontend" rule prevents. The
	// result panel reports what was dispatched instead of promising it up front.
	import {
		dispatchDownstreamStages,
		EnqueueRefused,
		type ConfigObject,
		type StageDispatchResult
	} from '$lib/api';

	let {
		config,
		blocker,
		maxAttempts,
		canExecute,
		runId = $bindable(''),
		onRefused
	}: {
		config: ConfigObject;
		blocker: { label: string; detail: string } | null;
		maxAttempts: number;
		canExecute: boolean;
		runId?: string;
		onRefused: () => void;
	} = $props();

	let dispatching = $state(false);
	let error: string | null = $state(null);
	let result = $state<StageDispatchResult | null>(null);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// The console writes rows and nothing else — execution stays on the workers,
	// so retry, crash recovery and one job list come along unchanged. `pii` and
	// `quality` are never dispatched; that bound is the server's
	// `DOWNSTREAM_STAGES`, so no press here can widen it.
	async function dispatch(): Promise<void> {
		if (dispatching) return;
		dispatching = true;
		error = null;
		result = null;
		try {
			result = await dispatchDownstreamStages({
				run_id: runId.trim(),
				config,
				max_attempts: maxAttempts
			});
		} catch (err) {
			if (err instanceof EnqueueRefused && err.status !== 400) onRefused();
			error = message(err);
		} finally {
			dispatching = false;
		}
	}
</script>

<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
	<div>
		<h2 class="font-display text-sm">Run downstream stages</h2>
		<p class="mt-1 max-w-prose text-xs text-muted-foreground">
			Dispatches the stages this config enables, in pipeline order, as queue jobs over
			the run's extracted shards — the same rows <code>womblex enqueue-stages</code>
			writes. Press it once extraction has drained on the
			<a href="/dashboard" class="underline">Dashboard</a>; a worker holds each stage
			until everything before it in the run has settled.
			<strong>PII masking and quality are never dispatched here</strong> — masking is
			irreversible, so it stays a deliberate <code>womblex run-stage</code> act.
			Pressing twice re-runs nothing.
		</p>
	</div>

	{#if blocker}
		<!-- Dispatch is off or unwired. Name the fix; the controls below stay
			 visible but disabled so the operator sees what they would do. -->
		<div class="rounded-md border border-status-warning/40 bg-status-warning/10 p-3">
			<p class="font-display text-xs">{blocker.label}</p>
			<p class="mt-1 max-w-prose text-xs text-muted-foreground">{blocker.detail}</p>
		</div>
	{/if}

	<div class="flex flex-wrap items-end gap-2 text-xs">
		<label class="flex flex-col gap-1">
			<span class="text-muted-foreground">Run id</span>
			<input
				type="text"
				bind:value={runId}
				placeholder="the run whose shards the stages read"
				disabled={!canExecute || dispatching}
				class="w-72 rounded-md border border-border bg-background px-2 py-1.5 font-mono
					focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
					disabled:opacity-50"
			/>
		</label>
		<button
			type="button"
			class="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground
				hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2
				focus-visible:ring-ring disabled:opacity-50 disabled:hover:bg-primary"
			disabled={!canExecute || dispatching || !runId.trim()}
			onclick={dispatch}
		>
			{dispatching ? 'Dispatching…' : 'Run downstream stages'}
		</button>
	</div>

	{#if error}
		<p class="text-xs text-status-failed">{error}</p>
	{/if}

	{#if result}
		<!-- What was actually dispatched, in the order the workers will claim it.
			 `newly_enqueued` reads 0 on a repeat press — the rows are idempotent per
			 (run_id, stage), so nothing already done is re-run. -->
		<dl
			class="flex flex-col gap-1 rounded-md border border-status-done/40 bg-status-done/10
				p-3 text-xs"
		>
			<div>
				<dt class="text-muted-foreground">Stages dispatched</dt>
				<dd class="font-mono">{result.stages.join(' → ')}</dd>
			</div>
			<div>
				<dt class="text-muted-foreground">Newly enqueued</dt>
				<dd class="font-mono">
					{result.newly_enqueued} of {result.stages.length}
					{result.newly_enqueued === result.stages.length
						? ''
						: '(the rest were already queued or done)'}
				</dd>
			</div>
			<div>
				<dt class="text-muted-foreground">Shards</dt>
				<dd class="truncate font-mono">{result.shard_prefix}</dd>
			</div>
		</dl>
	{/if}
</section>
