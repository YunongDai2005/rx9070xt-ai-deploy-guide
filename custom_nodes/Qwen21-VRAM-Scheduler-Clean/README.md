# Qwen21-VRAM-Scheduler-Clean

This replaces the mixed V1/V2/V3 scheduler files with one node.

Node:
- `Qwen 2.1 Scheduled Single-Image Edit`

Behavior:
- Prompt/image changed -> fully evict diffusion -> call official TextEncodeQwenImage21 -> fully unload text encoder -> sample.
- Seed-only changed -> conditioning cache is reused; encoder is not reloaded and diffusion can stay resident.

The node deliberately calls ComfyUI's official:
`comfy_extras.nodes_qwen.TextEncodeQwenImage21.execute(...)`

It passes the reference as:
`images={"image_1": image_1}`

Default reference resolution is 992.
