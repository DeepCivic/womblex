<script lang="ts">
	// The Pipeline Composer (docs/ui-plan.md merge 9). Like the Resources
	// Console it is not run-scoped: it edits configuration and shows the
	// pipeline's shape, so it never touches a run's artefacts.
	//
	// Both halves are served, not typed here: the graph is `STAGE_CONTRACTS`
	// and the form is `WomblexConfig`'s JSON Schema. Validation and the YAML
	// download go through the same `WomblexConfig(**raw)` construction
	// `load_config` uses, so the console cannot accept a config the CLI would
	// reject — and no guardrail is re-stated in TypeScript.
	import {
		getStageGraph,
		getConfigSchema,
		listPresets,
		savePreset,
		deletePreset,
		SavePresetRefused,
		validateConfig,
		renderConfigYaml,
		ConfigInvalid,
		getExecutionStatus,
		enqueueExtraction,
		EnqueueRefused,
		type ConfigObject,
		type ExecutionStatus,
		type EnqueueResult,
		type JsonSchema,
		type Preset,
		type StageGraph as StageGraphData,
		type ValidationResult
	} from '$lib/api';
	import { runSelection } from '$lib/stores/run.svelte';
	import StageGraph from '$lib/components/StageGraph.svelte';
	import SchemaForm, { defaultsFor } from '$lib/components/SchemaForm.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';

	// `$state<T | null>` rather than an annotation on the `let`: TypeScript
	// narrows the latter to `null` from its initialiser, which makes every
	// top-level `$derived` over it an error on `never`.
	let graph = $state<StageGraphData | null>(null);
	let schema = $state<JsonSchema | null>(null);
	let config: ConfigObject = $state({});
	let presets = $state<Preset[]>([]);
	let selectedPreset = $state('');
	let selected = $state<string | null>(null);
	let result = $state<ValidationResult | null>(null);
	let yamlText = $state<string | null>(null);
	let loading = $state(true);
	let busy = $state(false);
	let error = $state<string | null>(null);

	let defs = $derived(schema?.$defs ?? {});
	let selectedNode = $derived(graph?.nodes.find((n) => n.id === selected) ?? null);

	// Saving the composed config as a named preset (docs/ui-plan.md merge 9).
	// The server strips `dataset`/`paths` (a preset is an overlay) and validates
	// the rest, so this form sends the whole current config and lets the server
	// do the stripping — the browser re-implements no part of that.
	let saveName = $state('');
	let saveDescription = $state('');
	let saveFormats = $state('');
	let saving = $state(false);
	let saveError = $state<string | null>(null);
	// Whether saving is even offered. Set once the first save's 409 tells us this
	// console has no writable presets location (local mode, no --presets-dir);
	// remote mode always can, so this stays false there. Pre-emptive rather than
	// probed: the list endpoint does not report writability, so the first attempt
	// is what learns it, and the control then explains why it is disabled.
	let savingDisabled = $state(false);

	// Enqueue-from-composer (docs/ui-plan.md merge 11 hand-off). The queue carries
	// no config — workers get theirs from their own `--config` at launch — so
	// "enqueue this preset" is just handing the composed run's identity and input
	// location to the existing `enqueueExtraction`. `paths.input_root` seeds the
	// prefix but is only a suggestion: it may be an absolute/local path, whereas
	// `input_prefix` is store-relative, so the operator confirms it in a field of
	// its own rather than us assuming the two are the same.
	let execStatus = $state<ExecutionStatus | null>(null);
	let enqueuePrefix = $state('');
	let enqueueRunId = $state('');
	let enqueuing = $state(false);
	let enqueueError = $state<string | null>(null);
	let enqueueResult = $state<EnqueueResult | null>(null);

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	// A run id or path segment the composed config already carries, as a plain
	// string or empty — read defensively since `config` is a free-form object.
	function configString(section: string, key: string): string {
		const s = config[section];
		if (s === null || typeof s !== 'object' || Array.isArray(s)) return '';
		const v = (s as ConfigObject)[key];
		return typeof v === 'string' ? v : '';
	}

	$effect(() => {
		Promise.all([getStageGraph(), getConfigSchema(), listPresets(), getExecutionStatus()])
			.then(([g, s, p, x]) => {
				graph = g;
				schema = s;
				presets = p;
				execStatus = x;
				config = defaultsFor(s, s.$defs ?? {});
			})
			.catch((err) => (error = message(err)))
			.finally(() => (loading = false));
	});

	// Deep-merge a plain object overlay onto `base`, recursing into nested
	// objects (a stage section) and replacing scalars/arrays. Presets are
	// partial — they carry only the sections their shape sets — so a plain
	// spread would drop `base`'s other sections; and they never carry
	// `dataset`/`paths`, so those the operator already typed survive untouched.
	function deepMerge(base: ConfigObject, overlay: ConfigObject): ConfigObject {
		const out: ConfigObject = { ...base };
		for (const [key, value] of Object.entries(overlay)) {
			const existing = out[key];
			if (
				value !== null &&
				typeof value === 'object' &&
				!Array.isArray(value) &&
				existing !== null &&
				typeof existing === 'object' &&
				!Array.isArray(existing)
			) {
				out[key] = deepMerge(existing as ConfigObject, value as ConfigObject);
			} else {
				out[key] = structuredClone(value);
			}
		}
		return out;
	}

	// Loading a preset overlays its partial config onto the current form state
	// (schema defaults plus whatever the operator has set), so the run's
	// `dataset`/`paths` — which no preset carries — are preserved. Reassigning
	// `config` reruns the form; the reactive verdict clears via the same path an
	// edit takes.
	function applyPreset(name: string): void {
		const preset = presets.find((p) => p.name === name);
		if (!preset) return;
		config = deepMerge(config, preset.config);
		clearVerdict();
	}

	// Save the whole composed config as a named preset. `formats` is a space-
	// separated list of extensions; the server strips `dataset`/`paths` and
	// validates the overlay, so a 400 here is a name/overlay problem and a 409
	// means this console cannot write presets at all (then the control hides).
	async function save(): Promise<void> {
		if (saving || !saveName.trim()) return;
		saving = true;
		saveError = null;
		try {
			await savePreset({
				name: saveName.trim(),
				description: saveDescription.trim(),
				formats: saveFormats.trim() ? saveFormats.trim().split(/\s+/) : [],
				config
			});
			// Refresh the list so the new preset appears in the dropdown and, if
			// saved, becomes selectable/deletable straight away.
			presets = await listPresets();
			selectedPreset = saveName.trim();
			saveName = '';
			saveDescription = '';
			saveFormats = '';
		} catch (err) {
			if (err instanceof SavePresetRefused && err.status === 409) {
				savingDisabled = true;
			}
			saveError = message(err);
		} finally {
			saving = false;
		}
	}

	// Delete a saved preset (never a built-in — the control is only shown on
	// `source === 'saved'`). Refresh the list and clear the picker if it named
	// the one just removed.
	async function remove(name: string): Promise<void> {
		if (saving) return;
		saving = true;
		saveError = null;
		try {
			await deletePreset(name);
			presets = await listPresets();
			if (selectedPreset === name) selectedPreset = '';
		} catch (err) {
			saveError = message(err);
		} finally {
			saving = false;
		}
	}

	// Hand the composed run's identity and input location to the queue. The queue
	// carries no config; workers read their own `--config` at launch. So this is
	// only `paths.input_root` → `input_prefix` (confirmed, since it may be an
	// absolute/local path while the prefix is store-relative) and
	// `dataset.run_id` → `run_id`, through the same `enqueueExtraction` the
	// Execution Controls use.
	async function enqueue(): Promise<void> {
		if (enqueuing || !enqueuePrefix.trim()) return;
		enqueuing = true;
		enqueueError = null;
		enqueueResult = null;
		try {
			const res = await enqueueExtraction({
				input_prefix: enqueuePrefix.trim(),
				run_id: enqueueRunId.trim() || undefined
			});
			enqueueResult = res;
			// Point the rest of the console at the run just planned (as the
			// Execution Controls do), so the Dashboard tracks it.
			runSelection.select(res.run_id);
			await runSelection.load();
			runSelection.select(res.run_id);
		} catch (err) {
			// A capability change since load surfaces as 403/409; refresh status so
			// the control disables and explains, matching what the server saw.
			if (err instanceof EnqueueRefused && err.status !== 400) {
				getExecutionStatus()
					.then((body) => (execStatus = body))
					.catch(() => {});
			}
			enqueueError = message(err);
		} finally {
			enqueuing = false;
		}
	}

	// Any edit invalidates the last verdict: a stale green pill over a config
	// that has since changed is worse than no pill.
	function clearVerdict(): void {
		result = null;
		yamlText = null;
	}

	async function validate(): Promise<void> {
		busy = true;
		error = null;
		try {
			result = await validateConfig(config);
		} catch (err) {
			error = message(err);
		} finally {
			busy = false;
		}
	}

	async function generate(): Promise<void> {
		busy = true;
		error = null;
		try {
			yamlText = await renderConfigYaml(config);
			result = { valid: true, errors: [], unknown_keys: [] };
		} catch (err) {
			yamlText = null;
			if (err instanceof ConfigInvalid) {
				result = { valid: false, errors: err.errors, unknown_keys: [] };
			} else {
				error = message(err);
			}
		} finally {
			busy = false;
		}
	}

	// The rendered YAML is the server's, not the browser's — downloading it as
	// a blob of the text `/yaml` returned keeps the file byte-identical to what
	// `womblex run --config` would read.
	function download(): void {
		if (yamlText === null) return;
		const url = URL.createObjectURL(new Blob([yamlText], { type: 'application/yaml' }));
		const link = document.createElement('a');
		link.href = url;
		link.download = 'womblex.yaml';
		link.click();
		URL.revokeObjectURL(url);
	}

	const BUTTON =
		'rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-foreground/5 ' +
		'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50';
</script>

<div class="flex h-full flex-col gap-4 overflow-auto p-6">
	<h1 class="font-display text-2xl">Pipeline Composer</h1>

	{#if loading}
		<p class="text-sm text-muted-foreground">Loading pipeline shape…</p>
	{:else if error && !graph}
		<p class="text-sm text-status-failed">{error}</p>
	{:else if graph && schema}
		<!-- Preset picker: a named partial config the operator loads as a starting
			 point. It merges over the current form (preserving dataset/paths, which
			 no preset carries), so it seeds the four-stage shape without hand-
			 assembling it — e.g. DEFAULT-Isaacus's extract→chunk→enrich→build_graph
			 →money over PDF/DOCX. -->
		{#if presets.length > 0}
			{@const active = presets.find((p) => p.name === selectedPreset) ?? null}
			<section class="flex flex-col gap-2 rounded-md border border-border bg-surface-raised p-4">
				<label class="flex flex-wrap items-center gap-2 text-xs">
					<span class="font-display text-sm">Start from a preset</span>
					<select
						bind:value={selectedPreset}
						onchange={() => selectedPreset && applyPreset(selectedPreset)}
						class="rounded-md border border-border bg-background px-2 py-1 text-xs
							focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
					>
						<option value="">Custom (schema defaults)</option>
						{#each presets as preset (preset.name)}
							<option value={preset.name}>{preset.name}</option>
						{/each}
					</select>
					{#if active && active.source === 'saved'}
						<!-- Delete is offered only on operator-saved presets; a built-in is
							 code and 404s if deleted, so the control is not shown for it. -->
						<button
							type="button"
							class={BUTTON}
							disabled={saving}
							onclick={() => remove(active.name)}
						>
							Delete
						</button>
					{/if}
				</label>
				{#if active}
					<p class="max-w-prose text-xs text-muted-foreground">{active.description}</p>
					<p class="text-xs text-muted-foreground">
						Formats: <span class="font-mono">{active.formats.join(' ')}</span> ·
						<span class="font-mono">{active.source}</span> · sets stage toggles and settings only;
						your <code>dataset</code> and <code>paths</code> are kept.
					</p>
				{/if}

				<!-- Save the whole composed config as a named preset. The server strips
					 dataset/paths and validates the overlay; a 409 means this console
					 cannot write presets (local mode, no --presets-dir), and the control
					 then explains rather than reappearing to fail again. -->
				{#if !savingDisabled}
					<div class="mt-1 flex flex-col gap-2 border-t border-border pt-3">
						<span class="font-display text-sm">Save as preset</span>
						<div class="flex flex-wrap items-end gap-2 text-xs">
							<label class="flex flex-col gap-1">
								<span class="text-muted-foreground">Name</span>
								<input
									type="text"
									bind:value={saveName}
									placeholder="My-Run"
									class="rounded-md border border-border bg-background px-2 py-1 font-mono
										focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
								/>
							</label>
							<label class="flex flex-col gap-1">
								<span class="text-muted-foreground">Description <span class="opacity-60">(optional)</span></span>
								<input
									type="text"
									bind:value={saveDescription}
									placeholder="chunk only"
									class="rounded-md border border-border bg-background px-2 py-1
										focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
								/>
							</label>
							<label class="flex flex-col gap-1">
								<span class="text-muted-foreground">Formats <span class="opacity-60">(optional)</span></span>
								<input
									type="text"
									bind:value={saveFormats}
									placeholder=".pdf .docx"
									class="rounded-md border border-border bg-background px-2 py-1 font-mono
										focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
								/>
							</label>
							<button
								type="button"
								class={BUTTON}
								disabled={saving || !saveName.trim()}
								onclick={save}
							>
								{saving ? 'Saving…' : 'Save'}
							</button>
						</div>
						{#if saveError}
							<p class="text-xs text-status-failed">{saveError}</p>
						{/if}
					</div>
				{:else}
					<!-- 409: no writable presets location on this console. -->
					<p class="mt-1 border-t border-border pt-3 text-xs text-muted-foreground">
						Saving presets is disabled on this console — it has no writable presets
						location. Start it with <code>--presets-dir</code> (or
						<code>$WOMBLEX_UI_PRESETS_DIR</code>), or point it at a
						<code>--store</code> where presets go to the object store's own
						<code>presets/</code> prefix.
						{#if saveError}<span class="text-status-failed"> {saveError}</span>{/if}
					</p>
				{/if}
			</section>
		{/if}

		<!-- The DAG. Ordering is `required_inputs`, so "extraction precedes
			 chunking" is drawn from the contracts rather than asserted here. -->
		<StageGraph {graph} {config} bind:selected />

		{#if selectedNode}
			<section class="rounded-md border border-border bg-surface-raised p-4 text-xs">
				<div class="mb-2 flex flex-wrap items-center gap-2">
					<h2 class="font-display text-sm">{selectedNode.id}</h2>
					{#if selectedNode.needs_isaacus_api}
						<StatusPill status="warning" label="Isaacus API" />
					{/if}
					<span class="text-muted-foreground">
						{selectedNode.scope ?? 'in-batch'} · {selectedNode.mutation ?? 'extraction'}
						{#if selectedNode.config_section}· configured by <code>{selectedNode.config_section}</code>{/if}
					</span>
				</div>
				<dl class="grid gap-x-4 gap-y-1 sm:grid-cols-3">
					<dt class="text-muted-foreground">Reads</dt>
					<dt class="text-muted-foreground">Also reads (config-dependent)</dt>
					<dt class="text-muted-foreground">Writes</dt>
					<dd class="font-mono break-all">{selectedNode.required_inputs.join(' ') || '—'}</dd>
					<dd class="font-mono break-all">
						{selectedNode.conditional_inputs.map((c) => c.suffix).join(' ') || '—'}
					</dd>
					<dd class="font-mono break-all">{selectedNode.outputs.join(' ') || '—'}</dd>
				</dl>
			</section>
		{/if}

		<div class="grid gap-4 lg:grid-cols-[2fr_1fr]">
			<!-- The form. `onchange` on the wrapper catches every input inside it,
				 which is why the recursive form itself carries no verdict logic. -->
			<section
				class="flex flex-col gap-2 rounded-md border border-border bg-surface-raised p-4"
				onchange={clearVerdict}
				oninput={clearVerdict}
			>
				<h2 class="font-display text-sm">Configuration</h2>
				<SchemaForm {schema} {defs} value={config} />
			</section>

			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div class="flex flex-wrap items-center gap-2">
					<button type="button" class={BUTTON} onclick={validate} disabled={busy}>Validate</button>
					<button type="button" class={BUTTON} onclick={generate} disabled={busy}>
						Generate YAML
					</button>
					<button type="button" class={BUTTON} onclick={download} disabled={yamlText === null}>
						Download
					</button>
					{#if result}
						<StatusPill
							status={result.valid ? 'done' : 'failed'}
							label={result.valid ? 'Valid' : 'Invalid'}
						/>
					{/if}
				</div>

				{#if error}
					<p class="text-xs text-status-failed">{error}</p>
				{/if}

				{#if result && result.errors.length > 0}
					<ul class="flex flex-col gap-1 text-xs">
						{#each result.errors as err, i (i)}
							<li>
								<span class="font-mono">{err.loc.join('.') || '(root)'}</span>
								<span class="text-status-failed"> — {err.msg}</span>
							</li>
						{/each}
					</ul>
				{/if}

				{#if yamlText !== null}
					<pre
						class="max-h-96 overflow-auto rounded-md border border-border bg-surface-sunken p-3 font-mono text-[11px]">{yamlText}</pre>
				{/if}
			</section>
		</div>

		<!-- Enqueue this composed run (docs/ui-plan.md merge 11 hand-off). The queue
			 carries no config — workers read their own --config at launch — so this
			 hands off only the run's identity (dataset.run_id) and input location
			 (paths.input_root, as a *suggested* prefix: it may be absolute/local
			 while input_prefix is store-relative, so the operator confirms it).
			 Shown only where the console can dispatch (a store and a queue, not
			 --audit-only); otherwise the Execution Controls' banner names the fix. -->
		{#if execStatus?.can_execute}
			{@const suggestedPrefix = configString('paths', 'input_root')}
			{@const suggestedRunId = configString('dataset', 'run_id')}
			<section class="flex flex-col gap-3 rounded-md border border-border bg-surface-raised p-4">
				<div>
					<h2 class="font-display text-sm">Enqueue this run</h2>
					<p class="mt-1 max-w-prose text-xs text-muted-foreground">
						Plans an extraction run into the job queue from the composed run's identity
						and input location. Workers the platform brings up read their own
						<code>--config</code> at launch — the queue carries no config, so this hands
						off only the prefix and run id.
					</p>
				</div>
				<div class="flex flex-wrap items-end gap-2 text-xs">
					<label class="flex flex-col gap-1">
						<span class="text-muted-foreground">Input prefix</span>
						<input
							type="text"
							bind:value={enqueuePrefix}
							placeholder={suggestedPrefix || 'inbox'}
							disabled={enqueuing}
							class="rounded-md border border-border bg-background px-2 py-1.5 font-mono
								focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
								disabled:opacity-50"
						/>
					</label>
					<label class="flex flex-col gap-1">
						<span class="text-muted-foreground">Run id <span class="opacity-60">(optional)</span></span>
						<input
							type="text"
							bind:value={enqueueRunId}
							placeholder={suggestedRunId || 'mint a fresh timestamped id'}
							disabled={enqueuing}
							class="rounded-md border border-border bg-background px-2 py-1.5 font-mono
								focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
								disabled:opacity-50"
						/>
					</label>
					{#if suggestedPrefix && enqueuePrefix.trim() !== suggestedPrefix}
						<button
							type="button"
							class={BUTTON}
							disabled={enqueuing}
							onclick={() => {
								enqueuePrefix = suggestedPrefix;
								enqueueRunId = suggestedRunId;
							}}
						>
							Use composed paths
						</button>
					{/if}
					<button
						type="button"
						class="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground
							hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2
							focus-visible:ring-ring disabled:opacity-50 disabled:hover:bg-primary"
						disabled={enqueuing || !enqueuePrefix.trim()}
						onclick={enqueue}
					>
						{enqueuing ? 'Enqueuing…' : 'Enqueue run'}
					</button>
				</div>
				<p class="text-xs text-muted-foreground">
					<code>input_prefix</code> is store-relative; <code>paths.input_root</code> may be an
					absolute or local path, so confirm it rather than assuming they match.
				</p>

				{#if enqueueError}
					<p class="text-xs text-status-failed">{enqueueError}</p>
				{/if}

				{#if enqueueResult}
					<dl
						class="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-status-done/40
							bg-status-done/10 p-3 text-xs sm:grid-cols-4"
					>
						<div class="col-span-2 sm:col-span-4">
							<dt class="text-muted-foreground">Run id</dt>
							<dd class="font-mono">{enqueueResult.run_id}</dd>
						</div>
						<div>
							<dt class="text-muted-foreground">Documents</dt>
							<dd class="font-mono">{enqueueResult.document_count}</dd>
						</div>
						<div>
							<dt class="text-muted-foreground">Batches</dt>
							<dd class="font-mono">{enqueueResult.batch_count}</dd>
						</div>
						<div>
							<dt class="text-muted-foreground">Newly enqueued</dt>
							<dd class="font-mono">{enqueueResult.newly_enqueued}</dd>
						</div>
					</dl>
					<p class="text-xs text-muted-foreground">
						Selected this run — open the
						<a href="/dashboard" class="underline">Dashboard</a> to watch the queue drain.
					</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
