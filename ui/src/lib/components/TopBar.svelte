<script lang="ts">
	import RunSelector from '$lib/components/RunSelector.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
	import DensityToggle from '$lib/components/DensityToggle.svelte';
</script>

<header
	class="flex h-14 items-center gap-4 border-b border-border bg-surface-raised px-4"
>
	<a
		href="/"
		aria-label="Womblex console home"
		class="flex h-10 w-10 shrink-0 flex-col items-center justify-center rounded-md bg-foreground
			text-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
	>
		<span class="font-display text-[9px] leading-none tracking-wide">WOMB</span>
		<span class="font-display text-[9px] leading-none tracking-wide">LEX</span>
	</a>

	<input
		type="search"
		placeholder="Search…"
		aria-label="Global search"
		class="w-64 rounded-md border border-border bg-background px-3 py-1.5 text-sm
			placeholder:text-muted-foreground focus-visible:outline-none
			focus-visible:ring-2 focus-visible:ring-ring"
	/>

	<div class="flex-1"></div>

	<RunSelector />
	<DensityToggle />
	<ThemeToggle />
</header>
