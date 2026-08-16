<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import TopBar from '$lib/components/TopBar.svelte';
	import SideNav from '$lib/components/SideNav.svelte';
	import { preferences } from '$lib/stores/preferences.svelte';
	import { runSelection } from '$lib/stores/run.svelte';

	let { children } = $props();
	let navExpanded = $state(true);

	onMount(() => {
		runSelection.load();
	});

	// The token CSS keys off `:root[data-theme]` / `[data-density]`, so these
	// belong on <html> — not a wrapper div, which `:root` would never match.
	$effect(() => {
		document.documentElement.dataset.theme = preferences.theme;
		document.documentElement.dataset.density = preferences.density;
	});
</script>

<div class="flex h-screen flex-col">
	<TopBar />
	<div class="flex min-h-0 flex-1">
		<SideNav bind:expanded={navExpanded} />
		<main class="min-w-0 flex-1 overflow-auto">
			{@render children()}
		</main>
	</div>
</div>
