// Shared fetch plumbing for the console API client modules.

// Prefer the server's own `detail` (e.g. "run store unreachable: Install s3fs
// to access S3") over a bare status code, so a 503 from a misconfigured store
// reads as a fixable cause in the top banner rather than "GET /api/runs: 503".
export async function errorDetail(resp: Response, label: string): Promise<string> {
	const detail = ((await resp.json().catch(() => ({}))) as { detail?: unknown }).detail;
	if (typeof detail === 'string' && detail) return detail;
	return `${label}: ${resp.status}`;
}

// FastAPI's `detail` is a string for our raised HTTPExceptions but Pydantic's
// own list-of-errors for a 400 from a bad overlay — flatten both to one line.
export function detailToMessage(detail: unknown, status: number): string {
	if (typeof detail === 'string') return detail;
	if (Array.isArray(detail)) {
		return detail
			.map((e) => {
				const loc = Array.isArray(e?.loc) ? e.loc.join('.') : '';
				return loc ? `${loc}: ${e?.msg ?? ''}` : String(e?.msg ?? e);
			})
			.join('; ');
	}
	return `request failed: ${status}`;
}

export const JSON_POST = { method: 'POST', headers: { 'content-type': 'application/json' } };
