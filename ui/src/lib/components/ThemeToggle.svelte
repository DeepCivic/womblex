<script lang="ts">
	import { preferences, persistPreferences } from '$lib/stores/preferences.svelte';
	import Sun from '@lucide/svelte/icons/sun';
	import Moon from '@lucide/svelte/icons/moon';

	function toggle(): void {
		preferences.toggleTheme();
		persistPreferences();
	}
</script>

<button
	type="button"
	class="flex h-9 w-9 items-center justify-center rounded-md border border-border
		text-foreground hover:bg-surface-raised
		focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
	onclick={toggle}
	aria-label={preferences.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
>
	{#if preferences.theme === 'dark'}
		<Moon size={16} />
	{:else}
		<Sun size={16} />
	{/if}
</button>
