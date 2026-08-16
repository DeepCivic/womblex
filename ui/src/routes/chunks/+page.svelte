<script lang="ts">
	import ScreenStub from '$lib/components/ScreenStub.svelte';
</script>

<ScreenStub
	title="Chunk Inspector"
	reads="*.chunks.parquet, *.enrichment_entities.parquet, *.pii_spans.parquet, *.money_spans.parquet"
	merge={6}
/>
