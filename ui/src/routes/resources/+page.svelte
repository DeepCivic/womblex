<script lang="ts">
	import ScreenStub from '$lib/components/ScreenStub.svelte';
</script>

<ScreenStub
	title="Resources Console"
	reads="store/remote.storage_options_from_env(); isaacus_client.unserved_models(); JobQueue connectivity"
	merge={10}
/>
