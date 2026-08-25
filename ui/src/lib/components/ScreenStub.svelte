<script lang="ts">
	let { title, reads, merge }: { title: string; reads: string; merge: number } = $props();
</script>

<div class="flex h-full flex-col items-start gap-2 p-8">
	<h1 class="font-display text-2xl">{title}</h1>
	<p class="max-w-prose text-sm text-muted-foreground">
		<!-- No outbound link: the console is often air-gapped (DESIGN.md
		     adaptation principle 3), so a github.com href would be dead where
		     it matters most. `text-accent` is also unavailable as body copy —
		     DESIGN.md measures #7b6ff0 at 4.4:1, "Fails body", and this
		     confirms 4.42:1 dark / 3.64:1 light. -->
		Reads <code class="font-mono">{reads}</code>. Ships in
		<span class="text-foreground">delivery merge {merge}</span> — see
		<code class="font-mono">docs/ui-plan.md</code> §5.
	</p>
</div>
