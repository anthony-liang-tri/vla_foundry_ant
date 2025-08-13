import torch


def sample_chunk(input_ids, attention_mask, seq_len):
    if input_ids.shape[1] == seq_len + 1:
        start_idx = 0
    elif input_ids.shape[1] > seq_len + 1:
        start_idx = torch.randint(0, input_ids.shape[1] - seq_len, (1,)).item()
    else:
        seq_len = input_ids.shape[1]
        out_attention_mask = attention_mask[:, : seq_len - 1] if attention_mask is not None else None
        return input_ids[:, : seq_len - 1], out_attention_mask, input_ids[:, 1:seq_len]

    inputs = input_ids[:, start_idx : start_idx + seq_len]
    mask = attention_mask[:, start_idx : start_idx + seq_len] if attention_mask is not None else None
    targets = input_ids[:, start_idx + 1 : start_idx + seq_len + 1]
    return inputs, mask, targets
