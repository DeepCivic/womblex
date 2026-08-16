<script lang="ts">
	import ScreenStub from '$lib/components/ScreenStub.svelte';
</script>

<ScreenStub
	title="Dashboard"
	reads="JobQueue.stats(); per-stage CheckpointState; shard mtimes"
	merge={8}
/>
