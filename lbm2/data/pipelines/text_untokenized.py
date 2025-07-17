import webdataset as wds

from lbm2.data.pipelines.base import BaseWebDatasetPipeline
from lbm2.data.tokenizer import get_tokenizer
from lbm2.data.utils import deterministic_shuffle, log_and_continue


def batch_tokenize(batch, tokenizer, seq_len):
    texts = [item.decode("utf-8") if isinstance(item, bytes) else item for item in batch[0]]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len + 1,  # +1 because next token prediction
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


class TextUntokenizedPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality, data_configs, batch_size):
        super().__init__(modality, data_configs, batch_size)
        self.tokenizer = get_tokenizer(data_configs.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def create_pipeline(self, datastring, checkpoint_num):
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=0,
                initial=0,
                seed=self.data_configs.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.to_tuple("json", handler=log_and_continue),
            wds.batched(self.batch_size, partial=False),
            wds.map(self.tokenize_wrapper),
        ]
        return pipeline

    def tokenize_wrapper(self, batch):
        input_ids, attention_mask = batch_tokenize(batch, self.tokenizer, self.data_configs.seq_len)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
