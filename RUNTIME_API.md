# V7 runtime API

`AnimaNativeContextV7Runtime.generate_t2i(prompt, ...)` performs a hard
no-reference pass.

`AnimaNativeContextV7Runtime.generate(images, slots, prompt, ...)` accepts only:

- one reference with `slots=[0]`;
- two references with `slots=[0, 1]`;
- one Edit source with `slots=[0]`, `task_mode="edit"`, and
  `aligned_source_slot_id=0`.

Sparse or semantic single-image slots such as `[1]` are rejected. This is not a
loss of user control: a single active image has no second stream to distinguish.
The prompt remains unchanged and the model receives the source role through
structural metadata.

No inference method accepts a mask, a parser clause map, a LoRA path, or a
prompt-rewrite callback.
