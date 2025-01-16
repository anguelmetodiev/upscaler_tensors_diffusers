# Upscaler Helper

# Interrupt the diffusion process
def interrupt_callback(pipe, step_index, timestep, callback_kwargs):
    stop_idx = 15
    if step_index == stop_idx:
        pipe._interrupt = True
    return callback_kwargs

