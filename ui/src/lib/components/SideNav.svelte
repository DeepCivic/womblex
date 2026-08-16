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
						'flex items-center gap-3 rounded-md border-l-2 px-3 py-2 text-sm',
						active
							? 'border-l-primary text-primary'
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
	<button
		type="button"
		class="m-2 rounded-md p-2 text-xs text-muted-foreground hover:bg-surface-sunken hover:text-foreground"
		onclick={() => (expanded = !expanded)}
		aria-label={expanded ? 'Collapse navigation' : 'Expand navigation'}
	>
		{expanded ? '«' : '»'}
	</button>
</nav>
