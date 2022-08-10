import glob
import json
from tqdm import tqdm
from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)

from itertools import chain

def common_crawl_file_iterator(file):
    f = open(file, "r")
    items = f.readlines()
    for jsonl in items:
        doc = json.loads(jsonl)
        yield "".join(doc["cont"])


def batch_iterators(iterators):
    batches = []
    for iter in iterators:
        if len(batches) == 5:
            yield chain.from_iterable(batches)
            batches = []
            
        batches.append(iter)
    
    if len(batches) != 0:
        yield chain.from_iterable(batches)


def common_crawl(dir):
    # commoncrawl = "/mnt/cfs/commoncrawl-202*-**-s3-filter/"
    # dirs = ["/mnt/cfs/commoncrawl-2021-03-filter/minhash"]
    for file in glob.glob(dir + "/**.txt"):
        f = open(file, "r")
        items = f.readlines()
        for jsonl in items:
            doc = json.loads(jsonl)
            yield "".join(doc["cont"])


def weibo():
    # weibo
    weibo_path = "/mnt/cfs/weibo_comments/weibocomments_all.txt"
    f = open(weibo_path, "r")
    for conv in f.readlines():
        conv = json.loads(conv)
        # conv = [" ".join([w for w in c]) for c in conv["texts"]]
        # plain_conv = " <|endoftext|> ".join(conv)
        # yield plain_conv
        yield "\n".join(conv['texts'])


def lccc():
    # lccc
    lccc = "/mnt/cfs/LCCC/lccc_all.txt"
    f = open(lccc, "r")
    for conv in f.readlines():
        conv = json.loads(conv)
        yield "\n".join(conv['texts'])


def main():
    # # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # # Customize training
    # pre_tokenizer = pre_tokenizers.Sequence(
    # [pre_tokenizers.CharDelimiterSplit(" ")]
    # )
    # # pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=52000, special_tokens=[
                                  "<pad>", "<|endoftext|>", "<EMAIL>", "<PHONE>"])

    weibo_iterator = weibo()
    lccc_iterator = lccc()

    common_crawl_iterator = []
    # common_crawl_iterator = [common_crawl("/mnt/cfs/commoncrawl-2021-12-s3-filter/")]
    for file in glob.glob("/mnt/cfs/commoncrawl-2021-12-s3-filter/**.txt")[:200]:
        common_crawl_iterator.append(common_crawl_file_iterator(file))
    batch_common_crawl_iterator = batch_iterators(common_crawl_iterator)
    
    tokenizer.train_from_iterator(weibo_iterator, trainer=trainer)
    tokenizer.train_from_iterator(lccc_iterator, trainer=trainer)
    for iter in tqdm(batch_common_crawl_iterator):
        tokenizer.train_from_iterator(iter, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save("./tokenizer/gpt_bpe.json")


if __name__ == "__main__":
    main()
