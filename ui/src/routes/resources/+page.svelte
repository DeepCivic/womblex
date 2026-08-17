<script lang="ts">
	// The Resources Console (docs/ui-plan.md merge 10): three connection cards,
	// none of them run-scoped — like the Composer, this screen reads
	// deployment configuration, not a run's artefacts.
	import {
		getResources,
		testStoreConnection,
		testQueueConnection,
		type ResourcesCards,
		type ReachabilityResult,
		type QueueTestResult
	} from '$lib/api';
	import StatusPill from '$lib/components/StatusPill.svelte';

	let cards: ResourcesCards | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);

	let storeResult: ReachabilityResult | null = $state(null);
	let storeTesting = $state(false);
	let queueResult: QueueTestResult | null = $state(null);
	let queueTesting = $state(false);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// A hosted deployment with no API key is not "Configured" — it cannot make
	// a single call — but it has no unserved models either (that check only
	// means anything once endpoints are declared), so keying the pill on
	// coverage alone showed a green pill above an "API key: Not set" row.
	let isaacusState = $derived.by((): { status: 'done' | 'warning'; label: string } => {
		const card = cards?.isaacus;
		if (!card) return { status: 'warning', label: 'Unknown' };
		if (card.deployment === 'hosted' && !card.api_key_configured) {
			return { status: 'warning', label: 'No API key' };
		}
		if (card.unserved_models.length > 0) {
			return { status: 'warning', label: 'Missing coverage' };
		}
		return { status: 'done', label: 'Configured' };
	});

	$effect(() => {
		loading = true;
		error = null;
		getResources()
			.then((body) => (cards = body))
			.catch((err) => (error = message(err)))
			.finally(() => (loading = false));
	});

	async function runStoreTest(): Promise<void> {
		storeTesting = true;
		try {
			storeResult = await testStoreConnection();
		} catch (err) {
			storeResult = { reachable: false, error: message(err) };
		} finally {
			storeTesting = false;
		}
	}

	async function runQueueTest(): Promise<void> {
		queueTesting = true;
		try {
			queueResult = await testQueueConnection();
		} catch (err) {
			queueResult = { reachable: false, error: message(err), queue: null };
		} finally {
			queueTesting = false;
		}
	}
</script>

{#snippet testButton(
	label: string,
	testing: boolean,
	onclick: () => void,
	disabled: boolean = false
)}
	<button
		type="button"
		class="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-foreground/5
			focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50
			disabled:hover:bg-transparent"
		{onclick}
		disabled={testing || disabled}
	>
		{testing ? 'Testing…' : label}
	</button>
{/snippet}

{#snippet reachabilityPill(result: ReachabilityResult | null)}
	{#if result}
		<StatusPill
			status={result.reachable ? 'done' : 'failed'}
			label={result.reachable ? 'Reachable' : 'Unreachable'}
		/>
	{/if}
{/snippet}

<div class="flex h-full flex-col gap-4 p-6">
	<h1 class="font-display text-2xl">Resources Console</h1>

	{#if loading}
		<p class="text-sm text-muted-foreground">Loading connections…</p>
	{:else if error}
		<p class="text-sm text-status-failed">{error}</p>
	{:else if cards}
		<div class="grid grid-cols-1 gap-4 lg:grid-cols-3">
			<!-- Run store -->
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex items-center justify-between gap-2">
					<h2 class="font-display text-sm">Run store</h2>
					{@render reachabilityPill(storeResult)}
				</div>
				<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
					<dt class="text-muted-foreground">Kind</dt>
					<dd class="capitalize">{cards.store.kind}</dd>
					<dt class="text-muted-foreground">URI</dt>
					<dd class="truncate font-mono" title={cards.store.uri}>{cards.store.uri}</dd>
					{#if cards.store.kind === 'remote'}
						<dt class="text-muted-foreground">Object store</dt>
						<dd>{cards.store.is_object_store ? 'Yes' : 'No (local fsspec backend)'}</dd>
						<dt class="text-muted-foreground">Credentials</dt>
						<dd>{cards.store.options.credentials_configured ? 'Configured' : 'Not set'}</dd>
						{#if cards.store.options.endpoint_url}
							<dt class="text-muted-foreground">Endpoint</dt>
							<dd class="font-mono">{cards.store.options.endpoint_url}</dd>
						{/if}
						{#if cards.store.options.region}
							<dt class="text-muted-foreground">Region</dt>
							<dd>{cards.store.options.region}</dd>
						{/if}
					{/if}
				</dl>
				{#if storeResult?.error}
					<p class="text-xs text-status-failed">{storeResult.error}</p>
				{/if}
				{@render testButton('Test connection', storeTesting, runStoreTest)}
			</section>

			<!-- Job queue -->
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex items-center justify-between gap-2">
					<h2 class="font-display text-sm">Job queue</h2>
					{@render reachabilityPill(queueResult)}
				</div>
				<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
					<dt class="text-muted-foreground">Configured</dt>
					<dd>{cards.queue.configured ? 'Yes' : 'No — checkpoints only'}</dd>
					{#if cards.queue.dsn_masked}
						<dt class="text-muted-foreground">DSN</dt>
						<dd class="truncate font-mono" title={cards.queue.dsn_masked}
							>{cards.queue.dsn_masked}</dd
						>
					{/if}
				</dl>
				{#if queueResult?.error}
					<p class="text-xs text-status-failed">{queueResult.error}</p>
				{/if}
				{#if queueResult?.queue}
					<dl
						class="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-border bg-background p-3 text-xs sm:grid-cols-4"
					>
						{#each Object.entries(queueResult.queue.stats) as [status, count] (status)}
							<div>
								<dt class="text-muted-foreground capitalize">{status}</dt>
								<dd class="font-mono">{count}</dd>
							</div>
						{/each}
						<div>
							<dt class="text-muted-foreground">Total</dt>
							<dd class="font-mono">{queueResult.queue.total}</dd>
						</div>
						<div>
							<dt class="text-muted-foreground">Throughput</dt>
							<dd class="font-mono">{queueResult.queue.throughput.per_minute.toFixed(1)}/min</dd>
						</div>
					</dl>
					{#if queueResult.queue.workers.length > 0}
						<ul class="flex flex-col gap-1 text-xs">
							{#each queueResult.queue.workers as w (w.worker_id)}
								<li class="flex justify-between gap-2 font-mono">
									<span class="truncate">{w.worker_id}</span>
									<span class="shrink-0 text-muted-foreground">{w.running} running</span>
								</li>
							{/each}
						</ul>
					{:else}
						<p class="text-xs text-muted-foreground">No workers currently hold a lock.</p>
					{/if}
				{/if}
				<!-- Disabled without a DSN: the endpoint would answer "no queue
					 configured", which is what the card already says. A button whose
					 only outcome is to restate its own label is not an action. -->
				{@render testButton(
					cards.queue.configured ? 'Test connection' : 'No queue configured',
					queueTesting,
					runQueueTest,
					!cards.queue.configured
				)}
			</section>

			<!-- Isaacus -->
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex items-center justify-between gap-2">
					<h2 class="font-display text-sm">Isaacus</h2>
					<StatusPill status={isaacusState.status} label={isaacusState.label} />
				</div>
				<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
					<dt class="text-muted-foreground">Deployment</dt>
					<dd class="capitalize">{cards.isaacus.deployment}</dd>
					{#if cards.isaacus.deployment === 'hosted'}
						<dt class="text-muted-foreground">API key</dt>
						<dd class="font-mono"
							>{cards.isaacus.api_key_configured ? cards.isaacus.api_key_masked : 'Not set'}</dd
						>
					{/if}
				</dl>
				{#if cards.isaacus.endpoints.length > 0}
					<ul class="flex flex-col gap-1 text-xs">
						{#each cards.isaacus.endpoints as ep (ep.name)}
							<li class="font-mono">
								{ep.name}{ep.region ? `@${ep.region}` : ''} — {ep.models
									? ep.models.join('|')
									: 'all models'}
							</li>
						{/each}
					</ul>
				{/if}
				{#if cards.isaacus.unserved_models.length > 0}
					<p class="text-xs text-status-failed">
						No endpoint serves: {cards.isaacus.unserved_models.join(', ')}
					</p>
				{/if}
				<p class="text-xs text-muted-foreground">
					Checked against {cards.isaacus.models_checked.join(', ')}.
				</p>
			</section>
		</div>
	{/if}
</div>
