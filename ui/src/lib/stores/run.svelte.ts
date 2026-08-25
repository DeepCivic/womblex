import { listRuns, type RunSummary } from '$lib/api';

const SELECTED_RUN_KEY = 'womblex-console:selected-run';

class RunSelection {
	runs: RunSummary[] = $state([]);
	selectedRunId: string | null = $state(
		typeof localStorage === 'undefined' ? null : localStorage.getItem(SELECTED_RUN_KEY)
	);
	loading = $state(false);
	error: string | null = $state(null);

	async load(fetchImpl?: typeof fetch): Promise<void> {
		this.loading = true;
		this.error = null;
		try {
			this.runs = await listRuns(fetchImpl);
			if (!this.selectedRunId || !this.runs.some((r) => r.run_id === this.selectedRunId)) {
				this.select(this.runs[0]?.run_id ?? null);
			}
		} catch (err) {
			this.error = err instanceof Error ? err.message : String(err);
		} finally {
			this.loading = false;
		}
	}

	select(runId: string | null): void {
		this.selectedRunId = runId;
		if (typeof localStorage === 'undefined') return;
		if (runId) localStorage.setItem(SELECTED_RUN_KEY, runId);
		else localStorage.removeItem(SELECTED_RUN_KEY);
	}
}

export const runSelection = new RunSelection();
