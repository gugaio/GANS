def reshape_batch_to_1D(batch):
    assert batch.ndim > 1
    total_dim_size = batch.shape[1:].numel()
    return batch.reshape(-1, total_dim_size)