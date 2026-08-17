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
		validateConfig,
		renderConfigYaml,
		ConfigInvalid,
		type ConfigObject,
		type JsonSchema,
		type Preset,
		type StageGraph as StageGraphData,
		type ValidationResult
	} from '$lib/api';
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

	function message(err: unknown): string {
		return err instanceof Error ? err.message : String(err);
	}

	$effect(() => {
		Promise.all([getStageGraph(), getConfigSchema(), listPresets()])
			.then(([g, s, p]) => {
				graph = g;
				schema = s;
				presets = p;
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
				</label>
				{#if active}
					<p class="max-w-prose text-xs text-muted-foreground">{active.description}</p>
					<p class="text-xs text-muted-foreground">
						Formats: <span class="font-mono">{active.formats.join(' ')}</span> · sets stage toggles
						and settings only; your <code>dataset</code> and <code>paths</code> are kept.
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
	{/if}
</div>
