<script lang="ts">
	let { title, reads, merge }: { title: string; reads: string; merge: number } = $props();
</script>

<div class="flex h-full flex-col items-start gap-2 p-8">
	<h1 class="font-display text-2xl">{title}</h1>
	<p class="max-w-prose text-sm text-muted-foreground">
		Reads <code class="font-mono">{reads}</code>. Ships in
		<a
			href="https://github.com/DeepCivic/womblex/blob/main/docs/ui-plan.md"
			class="text-accent underline underline-offset-2"
		>
			delivery merge {merge}
		</a>
		 (docs/ui-plan.md §5).
	</p>
</div>
