<script lang="ts">
	// The Resources Console (docs/ui-plan.md merge 10; editable locations,
	// docs/ui-ingest-plan.md merge 3b): four connection cards, none of them
	// run-scoped — like the Composer, this screen reads deployment configuration,
	// not a run's artefacts. The ingest and output cards are the two the operator
	// can *edit* (env/compose values are defaults, not the only source); the
	// queue and Isaacus cards stay read-only and credential-masked.
	import {
		getResources,
		testStoreConnection,
		testIngest,
		testQueueConnection,
		saveLocations,
		getIngestPreflight,
		LocationsRefused,
		type ResourcesCards,
		type ReachabilityResult,
		type QueueTestResult,
		type IngestPreflight,
		type LocationSource
	} from '$lib/api';
	import StatusPill from '$lib/components/StatusPill.svelte';

	let cards: ResourcesCards | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);

	let storeResult: ReachabilityResult | null = $state(null);
	let storeTesting = $state(false);
	let ingestResult: ReachabilityResult | null = $state(null);
	let ingestTesting = $state(false);
	let queueResult: QueueTestResult | null = $state(null);
	let queueTesting = $state(false);

	// The document count on the ingest card, from the same preflight the composer
	// uses. Loaded once with the cards; refreshed after a successful ingest save.
	let preflight: IngestPreflight | null = $state(null);

	// Edit form state, one draft per editable card. Seeded from the effective URI
	// once the cards load, then the operator's to change.
	let ingestDraft = $state('');
	let storeDraft = $state('');
	let ingestSaving = $state(false);
	let storeSaving = $state(false);
	let ingestSaveError: string | null = $state(null);
	let storeSaveError: string | null = $state(null);
	// A 409 (no writable settings dir) or 403 (audit-only) is a fixed property of
	// this deployment, not a per-click failure: once seen, editing is disabled
	// permanently and the card explains the flag — the same pattern the composer
	// uses for preset saving. `cards.*.editable` already reports the 409 case up
	// front, so the controls never even render for it; this catches a 403 that
	// only the first save reveals.
	let locationsDisabled = $state(false);
	let locationsDisabledReason: string | null = $state(null);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// flag/env → "from environment"; saved → "set here" (docs/ui-ingest-plan.md
	// §2). The chip's colour follows: an operator-set value is the notable state.
	function provenance(source: LocationSource): { label: string; status: 'done' | 'skipped' } {
		return source === 'saved'
			? { label: 'set here', status: 'done' }
			: { label: 'from environment', status: 'skipped' };
	}

	// A hosted deployment with no API key is not "Configured" — it cannot make
	// a single call — but it has no unserved models either (that check only
	// means anything once endpoints are declared), so keying the pill on
	// coverage alone showed a green pill above an "API key: Not set" row.
	// A near-miss of the SageMaker endpoints var (e.g. the singular `…_ENDPOINT`)
	// is the usual cause of an unexpected "No API key", so it gets its own label.
	let isaacusState = $derived.by((): { status: 'done' | 'warning'; label: string } => {
		const card = cards?.isaacus;
		if (!card) return { status: 'warning', label: 'Unknown' };
		if (card.deployment === 'hosted' && card.endpoints_typo) {
			return { status: 'warning', label: 'Endpoints var misspelled' };
		}
		if (card.deployment === 'hosted' && !card.api_key_configured) {
			return { status: 'warning', label: 'No API key' };
		}
		if (card.unserved_models.length > 0) {
			return { status: 'warning', label: 'Missing coverage' };
		}
		return { status: 'done', label: 'Configured' };
	});

	// Seed the edit drafts from the effective locations, so the field opens
	// showing what is in force rather than blank. Set only once the cards land.
	function seedDrafts(body: ResourcesCards): void {
		ingestDraft = body.ingest.uri ?? '';
		storeDraft = body.store.uri;
	}

	$effect(() => {
		loading = true;
		error = null;
		Promise.all([getResources(), getIngestPreflight()])
			.then(([body, pf]) => {
				cards = body;
				preflight = pf;
				seedDrafts(body);
			})
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

	async function runIngestTest(): Promise<void> {
		ingestTesting = true;
		try {
			ingestResult = await testIngest();
		} catch (err) {
			ingestResult = { reachable: false, error: message(err) };
		} finally {
			ingestTesting = false;
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

	// A 409/403 is a deployment-wide state, not a per-field one, so it disables
	// *both* location cards' editing at once and explains the flag — the same
	// pattern the composer uses for preset saving. A 400 is per-value (bad URI,
	// overlap) and stays inline on the card that raised it.
	function handleRefusal(err: unknown): string {
		if (err instanceof LocationsRefused && (err.status === 409 || err.status === 403)) {
			locationsDisabled = true;
			locationsDisabledReason = err.message;
		}
		return message(err);
	}

	// Save the ingest override. `null` means reset to the flag/env default; a
	// non-empty draft is the new value. A PUT replaces the whole override, so the
	// store field is resubmitted at its current effective value to keep it.
	async function saveIngest(reset: boolean = false): Promise<void> {
		if (ingestSaving || !cards) return;
		ingestSaving = true;
		ingestSaveError = null;
		try {
			const result = await saveLocations({
				ingest_uri: reset ? null : ingestDraft.trim() || null,
				store_uri: cards.store.source === 'saved' ? cards.store.uri : null
			});
			cards = { ...cards, ingest: result.ingest, store: result.store };
			ingestResult = result.ingest_test;
			storeResult = result.store_test;
			seedDrafts(cards);
			preflight = await getIngestPreflight();
		} catch (err) {
			ingestSaveError = handleRefusal(err);
		} finally {
			ingestSaving = false;
		}
	}

	// Save the output override. Same shape as `saveIngest`: reset clears this
	// field, and the ingest field is resubmitted at its current value to keep it.
	async function saveStore(reset: boolean = false): Promise<void> {
		if (storeSaving || !cards) return;
		storeSaving = true;
		storeSaveError = null;
		try {
			const result = await saveLocations({
				store_uri: reset ? null : storeDraft.trim() || null,
				ingest_uri: cards.ingest.source === 'saved' ? cards.ingest.uri : null
			});
			cards = { ...cards, ingest: result.ingest, store: result.store };
			ingestResult = result.ingest_test;
			storeResult = result.store_test;
			seedDrafts(cards);
			preflight = await getIngestPreflight();
		} catch (err) {
			storeSaveError = handleRefusal(err);
		} finally {
			storeSaving = false;
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

<!-- The editable half of a location card: a text field, Save, Test and (only
	 once an override is in force) Reset-to-default. `editable` is false when this
	 console has no writable settings dir; `locationsDisabled` is a 409/403 that a
	 save revealed — either way the field disables and the flag is explained,
	 reusing the composer's preset-save pattern. -->
{#snippet locationEditor(
	value: string,
	set: (v: string) => void,
	source: LocationSource,
	editable: boolean,
	saving: boolean,
	saveError: string | null,
	onSave: () => void,
	onReset: () => void,
	onTest: () => void,
	testing: boolean
)}
	{#if editable && !locationsDisabled}
		<label class="flex flex-col gap-1 text-xs">
			<span class="text-muted-foreground">Location</span>
			<input
				type="text"
				{value}
				oninput={(e) => set(e.currentTarget.value)}
				placeholder="s3://womblex/inbox or /data/inbox"
				disabled={saving}
				class="rounded-md border border-border bg-background px-2 py-1.5 font-mono
					focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
			/>
		</label>
		<div class="flex flex-wrap items-center gap-2">
			<button
				type="button"
				class="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground
					hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
					disabled:opacity-50 disabled:hover:bg-primary"
				disabled={saving}
				onclick={onSave}
			>
				{saving ? 'Saving…' : 'Save'}
			</button>
			{@render testButton('Test connection', testing, onTest)}
			{#if source === 'saved'}
				<!-- Reset only where a saved override is in force — clearing the file
					 restores the flag/env default. -->
				{@render testButton('Reset to default', saving, onReset)}
			{/if}
		</div>
		{#if saveError}
			<p class="text-xs text-status-failed">{saveError}</p>
		{/if}
	{:else}
		{@render testButton('Test connection', testing, onTest)}
		<p class="text-xs text-muted-foreground">
			{#if locationsDisabled && locationsDisabledReason}
				{locationsDisabledReason}
			{:else}
				Location editing is disabled on this console — it has no writable settings
				dir. Start it with <code>--settings-dir</code> (or
				<code>$WOMBLEX_UI_SETTINGS_DIR</code>) to edit ingest and output locations.
			{/if}
		</p>
	{/if}
{/snippet}

<div class="flex h-full flex-col gap-4 p-6">
	<h1 class="font-display text-2xl">Resources Console</h1>

	{#if loading}
		<p class="text-sm text-muted-foreground">Loading connections…</p>
	{:else if error}
		<p class="text-sm text-status-failed">{error}</p>
	{:else if cards}
		<!-- The two editable location cards: where documents arrive from, and where
			 runs land. Env/compose values are defaults, not the only source — each
			 shows its provenance and can be overridden without a restart. -->
		<div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
			<!-- Ingest location -->
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex items-center justify-between gap-2">
					<h2 class="font-display text-sm">Ingest location</h2>
					<div class="flex items-center gap-2">
						{#if cards.ingest.configured}
							{@const p = provenance(cards.ingest.source)}
							<StatusPill status={p.status} label={p.label} />
						{/if}
						{@render reachabilityPill(ingestResult)}
					</div>
				</div>
				<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
					<dt class="text-muted-foreground">Configured</dt>
					<dd>{cards.ingest.configured ? 'Yes' : 'No — the composer cannot enqueue'}</dd>
					{#if cards.ingest.uri}
						<dt class="text-muted-foreground">URI</dt>
						<dd class="truncate font-mono" title={cards.ingest.uri}>{cards.ingest.uri}</dd>
						<dt class="text-muted-foreground">Object store</dt>
						<dd>{cards.ingest.is_object_store ? 'Yes' : 'No (local fsspec backend)'}</dd>
					{/if}
					{#if preflight?.reachable}
						<dt class="text-muted-foreground">Documents ready</dt>
						<dd class="font-mono">{preflight.document_count}</dd>
					{/if}
				</dl>
				{#if ingestResult?.error}
					<p class="text-xs text-status-failed">{ingestResult.error}</p>
				{:else if preflight && !preflight.reachable && preflight.error && cards.ingest.configured}
					<p class="text-xs text-status-failed">{preflight.error}</p>
				{/if}
				{@render locationEditor(
					ingestDraft,
					(v) => (ingestDraft = v),
					cards.ingest.source,
					cards.ingest.editable,
					ingestSaving,
					ingestSaveError,
					() => saveIngest(),
					() => saveIngest(true),
					runIngestTest,
					ingestTesting
				)}
			</section>

			<!-- Run store -->
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex items-center justify-between gap-2">
					<h2 class="font-display text-sm">Run store</h2>
					<div class="flex items-center gap-2">
						<StatusPill
							status={provenance(cards.store.source).status}
							label={provenance(cards.store.source).label}
						/>
						{@render reachabilityPill(storeResult)}
					</div>
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
				{@render locationEditor(
					storeDraft,
					(v) => (storeDraft = v),
					cards.store.source,
					cards.store.editable,
					storeSaving,
					storeSaveError,
					() => saveStore(),
					() => saveStore(true),
					runStoreTest,
					storeTesting
				)}
			</section>
		</div>

		<!-- The read-only connection cards. Credentials are env-provided and masked;
			 neither is editable here (docs/ui-ingest-plan.md §2 "accepting a credential
			 means storing one"). -->
		<div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
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
				{#if cards.isaacus.deployment === 'hosted' && cards.isaacus.endpoints_typo}
					<p class="text-xs text-status-failed">
						<code class="font-mono">{cards.isaacus.endpoints_typo}</code> is set, but Womblex
						reads <code class="font-mono">ISAACUS_SAGEMAKER_ENDPOINTS</code> (plural). Rename it,
						or the deployment falls back to the hosted API and reports no key.
					</p>
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
