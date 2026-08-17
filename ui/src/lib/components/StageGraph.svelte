<script lang="ts">
	// The pipeline DAG (docs/ui-plan.md merge 9), drawn from
	// `/api/composer/graph` — nodes and their ordering come from
	// `STAGE_CONTRACTS`, never from a hand-written list here.
	//
	// Nodes are HTML cards laid out at fixed pixel positions, with an SVG
	// behind them for the edges. Both read the same constants, so edge
	// geometry needs no DOM measurement.
	import type { StageGraph } from '$lib/api';

	let {
		graph,
		selected = $bindable(null)
	}: {
		graph: StageGraph;
		selected: string | null;
	} = $props();

	const NODE_W = 168;
	const NODE_H = 58;
	const GAP_X = 84;
	const GAP_Y = 12;

	// Longest path from a root: a stage sits one column right of its latest
	// dependency, so every edge points forward and none is drawn backwards.
	let layout = $derived.by(() => {
		const parents: Record<string, string[]> = {};
		for (const node of graph.nodes) parents[node.id] = [];
		for (const edge of graph.edges) parents[edge.to]?.push(edge.from);
		const depths: Record<string, number> = {};
		const depthOf = (id: string, seen: string[]): number => {
			if (depths[id] !== undefined) return depths[id];
			if (seen.includes(id)) return 0; // acyclic by contract; belt-and-braces
			const up = parents[id] ?? [];
			const depth = up.length ? Math.max(...up.map((p) => depthOf(p, [...seen, id]))) + 1 : 0;
			depths[id] = depth;
			return depth;
		};
		const rows: Record<number, number> = {};
		const at: Record<string, { x: number; y: number }> = {};
		for (const node of graph.nodes) {
			const depth = depthOf(node.id, []);
			const row = rows[depth] ?? 0;
			rows[depth] = row + 1;
			at[node.id] = { x: depth * (NODE_W + GAP_X), y: row * (NODE_H + GAP_Y) };
		}
		return {
			at,
			width: (Math.max(...Object.values(depths)) + 1) * (NODE_W + GAP_X) - GAP_X,
			height: Math.max(...Object.values(rows)) * (NODE_H + GAP_Y) - GAP_Y
		};
	});

	function edgePath(from: string, to: string): string {
		const a = layout.at[from];
		const b = layout.at[to];
		if (!a || !b) return '';
		const [x1, y1] = [a.x + NODE_W, a.y + NODE_H / 2];
		const [x2, y2] = [b.x, b.y + NODE_H / 2];
		const bend = Math.max(24, (x2 - x1) * 0.5);
		return `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`;
	}
</script>

<!-- `shrink-0`: the screen is a flex column, and without it the graph
     is squashed to whatever height is left over. -->
<div class="shrink-0 overflow-auto rounded-md border border-border bg-background p-4">
	<div class="relative" style="width:{layout.width}px;height:{layout.height}px">
		<svg class="absolute inset-0 text-accent" width={layout.width} height={layout.height}>
			{#each graph.edges as edge (`${edge.from}->${edge.to}`)}
				{@const touched = selected === edge.from || selected === edge.to}
				<path
					d={edgePath(edge.from, edge.to)}
					fill="none"
					stroke="currentColor"
					stroke-width={touched ? 2 : 1}
					opacity={selected && !touched ? 0.2 : 0.55}
				>
					<title>{edge.from} → {edge.to}: {edge.suffixes.join(', ')}</title>
				</path>
			{/each}
		</svg>

		{#each graph.nodes as node (node.id)}
			{@const pos = layout.at[node.id]}
			{#if pos}
				<div
					class={[
						'absolute flex flex-col justify-center gap-0.5 rounded-md border bg-surface-raised px-2.5',
						selected === node.id ? 'border-accent ring-1 ring-accent' : 'border-border'
					]}
					style="left:{pos.x}px;top:{pos.y}px;width:{NODE_W}px;height:{NODE_H}px"
				>
					<div class="flex items-center gap-2">
						<button
							type="button"
							class="truncate text-left font-display text-sm focus-visible:outline-none
								focus-visible:ring-2 focus-visible:ring-ring"
							onclick={() => (selected = selected === node.id ? null : node.id)}
						>
							{node.id}
						</button>
					</div>
					<p class="truncate text-[10px] text-muted-foreground">
						{node.required_inputs.length} in · {node.outputs.length} out{node.needs_isaacus_api
							? ' · Isaacus'
							: ''}
					</p>
				</div>
			{/if}
		{/each}
	</div>
</div>
