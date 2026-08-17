// The five domains (docs/ui-plan.md §3 "Screen → data source"). Order
// matches the delivery sequence so the rail reads top-to-bottom as the plan
// does; only the shell exists so far (merge 4) — each route is a stub until
// its own merge lands.
import Gauge from '@lucide/svelte/icons/gauge';
import FolderSearch from '@lucide/svelte/icons/folder-search';
import ScrollText from '@lucide/svelte/icons/scroll-text';
import Workflow from '@lucide/svelte/icons/workflow';
import Plug from '@lucide/svelte/icons/plug';
import Play from '@lucide/svelte/icons/play';
import type { Component } from 'svelte';

export interface NavItem {
	label: string;
	href: string;
	icon: Component<{ size?: number | string }>;
}

export const NAV_ITEMS: NavItem[] = [
	{ label: 'Dashboard', href: '/dashboard', icon: Gauge },
	{ label: 'Corpus Inspector', href: '/corpus', icon: FolderSearch },
	{ label: 'Chunk Inspector', href: '/chunks', icon: ScrollText },
	{ label: 'Pipeline Composer', href: '/composer', icon: Workflow },
	{ label: 'Resources Console', href: '/resources', icon: Plug },
	{ label: 'Execution Controls', href: '/execute', icon: Play }
];
