// Thin fetch wrapper over the read API `womblex.ui.routes` serves
// (docs/ui-plan.md §3). No client-side caching or state management beyond
// what each screen needs — the console reads shard artefacts, it doesn't
// mutate them, so there is nothing to keep in sync.

export interface RunSummary {
	run_id: string;
	document_count: number;
	stages: string[];
	created_at: string | null;
	updated_at: string | null;
}

export async function listRuns(fetchImpl: typeof fetch = fetch): Promise<RunSummary[]> {
	const resp = await fetchImpl('/api/runs');
	if (!resp.ok) throw new Error(`GET /api/runs: ${resp.status}`);
	const body = (await resp.json()) as { runs: RunSummary[] };
	return body.runs;
}
