<script lang="ts">
	import ScreenStub from '$lib/components/ScreenStub.svelte';
</script>

<ScreenStub
	title="Corpus Inspector"
	reads="<run_id>/manifest.parquet; shard_audit.audit_shard_directory()"
	merge={5}
/>
