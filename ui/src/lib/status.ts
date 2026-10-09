// One icon + fill per status value (DESIGN.md "Component conventions" —
// `StatusPill`), reused wherever a status needs rendering.
import CircleCheck from '@lucide/svelte/icons/circle-check';
import CircleX from '@lucide/svelte/icons/circle-x';
import Clock from '@lucide/svelte/icons/clock';
import Loader from '@lucide/svelte/icons/loader';
import AlertTriangle from '@lucide/svelte/icons/alert-triangle';
import SkipForward from '@lucide/svelte/icons/skip-forward';
import type { Component } from 'svelte';

export type Status = 'pending' | 'running' | 'done' | 'failed' | 'warning' | 'skipped';

export const STATUS_ICON: Record<Status, Component<{ size?: number | string }>> = {
	pending: Clock,
	running: Loader,
	done: CircleCheck,
	failed: CircleX,
	warning: AlertTriangle,
	skipped: SkipForward
};

export const STATUS_BG: Record<Status, string> = {
	pending: 'bg-status-pending',
	running: 'bg-status-running',
	done: 'bg-status-done',
	failed: 'bg-status-failed',
	warning: 'bg-status-warning',
	// Reuses the pending fill — legal because the icon, not the colour,
	// carries the distinction (DESIGN.md).
	skipped: 'bg-status-pending'
};

// Text on each fill. Light fills take white and dark fills near-black
// (`--status-foreground` flips per theme); the lime running fill takes
// near-black in both.
export const STATUS_FG: Record<Status, string> = {
	pending: 'text-status-foreground',
	running: 'text-status-running-foreground',
	done: 'text-status-foreground',
	failed: 'text-status-foreground',
	warning: 'text-status-foreground',
	skipped: 'text-status-foreground'
};
