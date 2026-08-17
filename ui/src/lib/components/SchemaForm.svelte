<script module lang="ts">
	// A form over one Pydantic model's JSON Schema (docs/ui-plan.md merge 9).
	// Recursive: a `$ref` property renders as a collapsible subsection, so the
	// whole of `WomblexConfig` is reachable without a hand-typed mirror of
	// `config.py` — the field list is whatever `/api/composer/schema` serves.
	import type { ConfigObject, JsonSchema } from '$lib/api';

	export type FieldKind = 'boolean' | 'number' | 'string' | 'strings' | 'mapping' | 'model';

	export interface Field {
		kind: FieldKind;
		nullable: boolean;
		model: JsonSchema | null;
	}

	function deref(schema: JsonSchema, defs: Record<string, JsonSchema>): JsonSchema {
		const ref = schema.$ref?.split('/').pop();
		return ref ? (defs[ref] ?? schema) : schema;
	}

	/** How to edit *prop*: its kind, whether null is allowed, and (for a nested
	 * model) the schema to recurse into. `X | None` is read through — the union
	 * is Pydantic's optionality marker, not a variant to render a picker for. */
	export function fieldOf(prop: JsonSchema, defs: Record<string, JsonSchema>): Field {
		const branches = prop.anyOf ?? [prop];
		const nullable = branches.some((b) => b.type === 'null');
		let field: Field = { kind: 'string', nullable, model: null };
		for (const raw of branches) {
			if (raw.type === 'null') continue;
			const branch = deref(raw, defs);
			if (branch.properties) field = { ...field, kind: 'model', model: branch };
			else if (branch.type === 'array') field = { ...field, kind: 'strings' };
			else if (branch.type === 'object') field = { ...field, kind: 'mapping' };
			else if (branch.type === 'boolean') field = { ...field, kind: 'boolean' };
			else if (branch.type === 'integer' || branch.type === 'number')
				field = { ...field, kind: 'number' };
			// `int | float` is two numeric branches; either lands on 'number'.
		}
		return field;
	}

	/** *schema* at its defaults — what `WomblexConfig()` would hold. Required
	 * fields have no default (`dataset.name`, the paths), so they start empty:
	 * Pydantic accepts an empty string for each, so the composer shows the
	 * field rather than inventing a default the library never had. */
	export function defaultsFor(
		schema: JsonSchema,
		defs: Record<string, JsonSchema>
	): ConfigObject {
		const out: ConfigObject = {};
		for (const [key, prop] of Object.entries(schema.properties ?? {})) {
			const field = fieldOf(prop, defs);
			if (field.kind === 'model' && field.model) {
				// Merge under the model's own defaults: a section default is a
				// full dump today, but a partial one would otherwise leave its
				// missing keys bound to `undefined`.
				out[key] = {
					...defaultsFor(field.model, defs),
					...((prop.default as ConfigObject | undefined) ?? {})
				};
			} else if (prop.default !== undefined) {
				out[key] = structuredClone(prop.default);
			} else {
				out[key] = field.nullable ? null : field.kind === 'boolean' ? false : '';
			}
		}
		return out;
	}
</script>

<script lang="ts">
	import Self from './SchemaForm.svelte';

	let {
		schema,
		defs,
		value
	}: {
		schema: JsonSchema;
		defs: Record<string, JsonSchema>;
		/** Mutated in place — a `$state` proxy from the screen, so nested edits
		 * reach the config the Validate/YAML actions post. */
		value: ConfigObject;
	} = $props();

	// Free-form mappings (`engine_options`, `substitutions`) are edited as JSON
	// text. The draft is kept per field so a half-typed object neither reaches
	// the config nor gets overwritten mid-keystroke.
	let drafts: Record<string, string> = $state({});
	let badJson: Record<string, boolean> = $state({});

	let entries = $derived(Object.entries(schema.properties ?? {}));

	function editMapping(key: string, text: string): void {
		drafts[key] = text;
		try {
			value[key] = JSON.parse(text || '{}');
			badJson[key] = false;
		} catch {
			badJson[key] = true;
		}
	}

	function isObject(value: unknown): boolean {
		return typeof value === 'object' && value !== null && !Array.isArray(value);
	}

	const CLEAR = 'mt-2 text-xs text-muted-foreground underline hover:text-foreground';
	const INPUT =
		'w-full rounded-md border border-border bg-background px-2 py-1 font-mono text-xs ' +
		'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';
</script>

<div class="flex flex-col gap-1.5">
	{#each entries as [key, prop] (key)}
		{@const field = fieldOf(prop, defs)}
		{#if field.kind === 'model' && field.model}
			{@const model = field.model}
			{@const set = isObject(value[key])}
			<details class="rounded-md border border-border" open={schema.required?.includes(key)}>
				<summary class="cursor-pointer px-2 py-1.5 text-xs font-semibold">{key}</summary>
				<div class="border-t border-border p-2">
					<!-- An optional subsection defaulting to null (`linking.reference`)
						 stays null until the operator asks for it: seeding it here
						 would post a section Pydantic never had, and doing it from a
						 template expression is a state mutation Svelte rejects. -->
					{#if set}
						<Self schema={model} {defs} value={value[key] as ConfigObject} />
						{#if field.nullable}
							<button type="button" class={CLEAR} onclick={() => (value[key] = null)}>
								Clear section
							</button>
						{/if}
					{:else}
						<button
							type="button"
							class={CLEAR}
							onclick={() => (value[key] = defaultsFor(model, defs))}
						>
							Not set — configure {key}
						</button>
					{/if}
				</div>
			</details>
		{:else}
			<label class="grid grid-cols-[minmax(0,11rem)_1fr] items-center gap-3">
				<span class="truncate text-xs text-muted-foreground" title={prop.description ?? key}>
					{key}
				</span>
				{#if field.kind === 'boolean'}
					<input
						type="checkbox"
						class="h-4 w-4 justify-self-start accent-accent"
						checked={value[key] === true}
						onchange={(e) => (value[key] = e.currentTarget.checked)}
					/>
				{:else if field.kind === 'number'}
					<input
						type="number"
						step="any"
						class={INPUT}
						value={(value[key] ?? '') as number | string}
						oninput={(e) =>
							(value[key] = e.currentTarget.value === '' ? null : e.currentTarget.valueAsNumber)}
					/>
				{:else if field.kind === 'strings'}
					<input
						class={INPUT}
						placeholder="comma-separated"
						value={Array.isArray(value[key]) ? (value[key] as string[]).join(', ') : ''}
						oninput={(e) =>
							(value[key] = e.currentTarget.value
								.split(',')
								.map((s) => s.trim())
								.filter(Boolean))}
					/>
				{:else if field.kind === 'mapping'}
					<input
						class={[INPUT, badJson[key] && 'border-status-failed']}
						placeholder="JSON object"
						value={drafts[key] ?? JSON.stringify(value[key] ?? {})}
						oninput={(e) => editMapping(key, e.currentTarget.value)}
					/>
				{:else}
					<input
						class={INPUT}
						value={(value[key] ?? '') as string}
						oninput={(e) =>
							(value[key] =
								e.currentTarget.value === '' && field.nullable ? null : e.currentTarget.value)}
					/>
				{/if}
			</label>
		{/if}
	{/each}
</div>
