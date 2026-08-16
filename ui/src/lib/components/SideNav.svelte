<script lang="ts">
	import { page } from '$app/state';
	import { NAV_ITEMS } from '$lib/nav';

	let { expanded = $bindable(true) }: { expanded?: boolean } = $props();
</script>

<nav
	class="flex h-full flex-col border-r border-border bg-surface-raised transition-[width] duration-300"
	style:width={expanded ? '224px' : '64px'}
	aria-label="Console sections"
>
	<ul class="flex flex-1 flex-col gap-1 p-2">
		{#each NAV_ITEMS as item (item.href)}
			{@const active = page.url.pathname.startsWith(item.href)}
			<li>
				<a
					href={item.href}
					class={[
						// Height comes from --row-height so the density control has a
						// real consumer in the shell, rather than only taking effect
						// once the grids land.
						'flex h-[var(--row-height)] items-center gap-3 rounded-md border-l-2 px-3 text-sm',
						'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
						// Lime marks the active item as the 2px *rule* only. As label
						// or icon colour it measures 11.9:1 on the dark nav but 1.3:1
						// on the light one (--surface-raised is #ffffff there), so the
						// text carries the active state with weight instead — legible
						// at 13.7:1 dark / 17.5:1 light.
						active
							? 'border-l-primary font-semibold text-foreground'
							: 'border-l-transparent text-muted-foreground hover:text-foreground'
					]}
					aria-current={active ? 'page' : undefined}
				>
					<item.icon size={18} />
					{#if expanded}
						<span class="truncate">{item.label}</span>
					{/if}
				</a>
			</li>
		{/each}
	</ul>
	<!-- hover:bg-background, not surface-sunken: the nav is already
	     surface-raised, and sunken is the "well" token (lime in light mode
	     per DESIGN.md), not a hover state. -->
	<button
		type="button"
		class="m-2 rounded-md p-2 text-xs text-muted-foreground hover:bg-background hover:text-foreground
			focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
		onclick={() => (expanded = !expanded)}
		aria-label={expanded ? 'Collapse navigation' : 'Expand navigation'}
	>
		{expanded ? '«' : '»'}
	</button>
</nav>
