import { detailToMessage, errorDetail, JSON_POST } from './http';

// The Pipeline Composer (functional_requirements.md, Requirement 19). Neither read is run-scoped:
// the DAG comes from `STAGE_CONTRACTS` and the form's fields from
// `WomblexConfig`'s JSON Schema, so the frontend hand-codes neither the stage
// ordering nor the config field list.
export interface ConditionalInput {
	suffix: string;
	reason: string;
	strict: boolean;
}

export interface StageNode {
	id: string;
	scope: string | null;
	mutation: string | null;
	needs_isaacus_api: boolean;
	checkpoint_dirname: string | null;
	required_inputs: string[];
	conditional_inputs: ConditionalInput[];
	outputs: string[];
	// The `WomblexConfig` section that configures this stage, or null where no
	// single section does (`extract`, `graph-refresh`). Served rather than
	// mapped here so a renamed config section breaks a Python test.
	config_section: string | null;
}

export interface StageEdge {
	from: string;
	to: string;
	suffixes: string[];
}

export interface StageGraph {
	nodes: StageNode[];
	edges: StageEdge[];
}

export async function getStageGraph(fetchImpl: typeof fetch = fetch): Promise<StageGraph> {
	const resp = await fetchImpl('/api/composer/graph');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/composer/graph'));
	return (await resp.json()) as StageGraph;
}

// A named pre-configured pipeline the composer form loads as a starting point
// (e.g. `DEFAULT-Isaacus`: extract → chunk → enrich → build_graph → money).
// `config` is a *partial* `WomblexConfig` — it carries stage toggles/settings
// but never `dataset`/`paths`, which name the run and stay the operator's to
// fill — so applying it merges over the form's current config, leaving those.
// `source` is `builtin` (code, undeletable) or `saved` (operator-authored) —
// the screen offers delete only on the latter.
export interface Preset {
	name: string;
	description: string;
	formats: string[];
	config: ConfigObject;
	source: 'builtin' | 'saved';
}

export async function listPresets(fetchImpl: typeof fetch = fetch): Promise<Preset[]> {
	const resp = await fetchImpl('/api/composer/presets');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/composer/presets'));
	const body = (await resp.json()) as { presets: Preset[] };
	return body.presets;
}

// The save-a-preset form. `config` is the whole composed config; the server
// strips `dataset`/`paths` (a preset is an overlay) and validates the rest
// through the same `WomblexConfig(**raw)` the composer's other endpoints use.
export interface SavePresetRequest {
	name: string;
	description?: string;
	formats?: string[];
	config: ConfigObject;
}

/**
 * Saving a preset the console refused. `status` is the HTTP code so the screen
 * distinguishes the failure shapes without parsing the message: 409 (this
 * console cannot write presets — local mode with no presets dir; remote mode
 * always can), 400 (unsafe name, or a config that would not load as an
 * overlay). Mirrors `ui/routes/composer.py`'s guard and error paths.
 */
export class SavePresetRefused extends Error {
	status: number;
	constructor(status: number, detail: string) {
		super(detail);
		this.name = 'SavePresetRefused';
		this.status = status;
	}
}

export async function savePreset(
	req: SavePresetRequest,
	fetchImpl: typeof fetch = fetch
): Promise<Preset> {
	const resp = await fetchImpl('/api/composer/presets', {
		...JSON_POST,
		body: JSON.stringify(req)
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		throw new SavePresetRefused(resp.status, detailToMessage(detail, resp.status));
	}
	return (await resp.json()) as Preset;
}

export async function deletePreset(
	name: string,
	fetchImpl: typeof fetch = fetch
): Promise<void> {
	const resp = await fetchImpl(`/api/composer/presets/${encodeURIComponent(name)}`, {
		method: 'DELETE'
	});
	if (!resp.ok) {
		const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
		throw new SavePresetRefused(resp.status, detailToMessage(detail, resp.status));
	}
}

// The subset of JSON Schema Pydantic emits for `WomblexConfig`. Typed rather
// than `any` so the form's field resolution is checked; unknown keywords are
// simply not read.
export interface JsonSchema {
	type?: string;
	title?: string;
	description?: string;
	default?: unknown;
	properties?: Record<string, JsonSchema>;
	required?: string[];
	items?: JsonSchema;
	anyOf?: JsonSchema[];
	enum?: unknown[];
	additionalProperties?: boolean | JsonSchema;
	$ref?: string;
	$defs?: Record<string, JsonSchema>;
}

export type ConfigObject = Record<string, unknown>;

export async function getConfigSchema(fetchImpl: typeof fetch = fetch): Promise<JsonSchema> {
	const resp = await fetchImpl('/api/composer/schema');
	if (!resp.ok) throw new Error(await errorDetail(resp, 'GET /api/composer/schema'));
	return (await resp.json()) as JsonSchema;
}

// Pydantic's own error shape (`include_url`/`context`/`input` off server-side).
export interface ConfigError {
	loc: (string | number)[];
	msg: string;
	type: string;
}

export interface ValidationResult {
	valid: boolean;
	errors: ConfigError[];
}

export async function validateConfig(
	raw: ConfigObject,
	fetchImpl: typeof fetch = fetch
): Promise<ValidationResult> {
	const resp = await fetchImpl('/api/composer/validate', { ...JSON_POST, body: JSON.stringify(raw) });
	if (!resp.ok) throw new Error(await errorDetail(resp, 'POST /api/composer/validate'));
	return (await resp.json()) as ValidationResult;
}

/** A config `/yaml` refused to render — carries the same errors `/validate` reports. */
export class ConfigInvalid extends Error {
	errors: ConfigError[];
	constructor(errors: ConfigError[]) {
		super('Config is invalid');
		this.name = 'ConfigInvalid';
		this.errors = errors;
	}
}

export async function renderConfigYaml(
	raw: ConfigObject,
	fetchImpl: typeof fetch = fetch
): Promise<string> {
	const resp = await fetchImpl('/api/composer/yaml', { ...JSON_POST, body: JSON.stringify(raw) });
	if (resp.status === 422) {
		throw new ConfigInvalid(((await resp.json()) as { detail: ConfigError[] }).detail);
	}
	if (!resp.ok) throw new Error(await errorDetail(resp, 'POST /api/composer/yaml'));
	return await resp.text();
}
