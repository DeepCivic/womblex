<script lang="ts">
	import { STATUS_ICON, STATUS_BG, STATUS_FG, type Status } from '$lib/status';

	let { status, label }: { status: Status; label?: string } = $props();
	let Icon = $derived(STATUS_ICON[status]);
</script>

<span
	class={[
		'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
		STATUS_BG[status],
		STATUS_FG[status]
	]}
>
	<span aria-hidden="true"><Icon size={12} /></span>
	{label ?? status}
</span>
