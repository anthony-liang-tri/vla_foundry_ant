import webdataset as wds
from data.pipelines.base import BaseWebDatasetPipeline
from data.utils import deterministic_shuffle, log_and_continue


def filter_lt_seqlen(seq_len, x):
    valid_sample = len(x) > seq_len
    if not valid_sample:
        logging.warning(
            f"Sample sequence length: {len(x)} not larger than seq_len: {seq_len}. Skipping sample. NOTE: sample sequence length should be one greater than seq_len."
        )
    return valid_sample


class TextPipeline(BaseWebDatasetPipeline):
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
            wds.decode(handler=log_and_continue),
            # wds.to_tuple("json.gz", handler=log_and_continue),
            wds.map(lambda sample: {"input_ids": sample["json.gz"]}, handler=log_and_continue),
            wds.select(lambda x: filter_lt_seqlen(self.data_configs.seq_len, x["input_ids"])),
            wds.batched(self.batch_size, partial=False),
        ]
        return pipeline