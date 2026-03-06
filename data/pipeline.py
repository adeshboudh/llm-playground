# data/pipeline.py

import yaml
from data.ingestion.downloader import DatasetDownloader, DownloaderConfig
from data.cleaning.filters import (
    FilterPipeline, LengthFilter, WordLengthFilter,
    SymbolRatioFilter, BulletLinesFilter, AlphanumericFilter
)
from data.cleaning.deduplicator import MinHashDeduplicator
from data.tokenizer.trainer import BPETokenizerTrainer
from data.tokenizer.encoder import ShardEncoder
from data.registry.hub_pusher import HFHubPusher

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def run_pipeline(config_path: str, token: str = os.environ["HF_TOKEN"]):
    # ── Load Config ──────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ic = cfg["ingestion"]
    cc = cfg["cleaning"]
    tc = cfg["tokenizer"]
    rc = cfg["registry"]

    # ── Step 1: Downloader ────────────────────────────
    logger.info("=== STEP 1: Data Ingestion ===")
    downloader = DatasetDownloader(DownloaderConfig(
        source=ic["source"],
        subset=ic["subset"],
        num_samples=ic["num_samples"],
        text_column=ic["text_column"],
        streaming=ic["streaming"]
    )).load()

    # ── Step 2: Filter + Dedup (Single Pass) ─────────
    logger.info("=== STEP 2: Filtering + Dedup ===")
    filter_pipeline = FilterPipeline([
        LengthFilter(cc["min_length"], cc["max_length"]),
        WordLengthFilter(cc["min_avg_word_length"], cc["max_avg_word_length"]),
        SymbolRatioFilter(cc["max_symbol_to_word_ratio"]),
        BulletLinesFilter(cc["max_bullet_lines_ratio"]),
        AlphanumericFilter()
    ])
    deduplicator = MinHashDeduplicator(
        num_perm=cc["minhash_num_perm"],
        threshold=cc["minhash_threshold"]
    )

    clean_docs = []
    for i, text in enumerate(downloader.iterate()):
        if not filter_pipeline.apply(text):
            continue
        if deduplicator.is_duplicate(text, doc_id=str(i)):
            continue
        clean_docs.append(text)

    logger.info(f"Clean docs after filter+dedup: {len(clean_docs)}")

    # ── Step 3: Train Tokenizer ───────────────────────
    logger.info("=== STEP 3: Tokenizer Training ===")
    trainer = BPETokenizerTrainer(
        vocab_size=tc["vocab_size"],
        min_frequency=tc["min_frequency"],
        special_tokens=tc["special_tokens"]
    ).train(iter(clean_docs))
    saved_tokenizer_path = trainer.save(cfg["artifacts"]["tokenizer_path"])

    # ── Step 4: Encode Shards ─────────────────────────
    logger.info("=== STEP 4: Shard Encoding ===")
    encoder = ShardEncoder(
        tokenizer=trainer,
        output_dir=cfg["artifacts"]["shards_dir"],
        shard_size=tc["shard_size_tokens"]
    )
    stats = encoder.encode_stream(iter(clean_docs))
    logger.info(f"Encoding complete: {stats}")


    # ── Step 5: Push to HF Hub ────────────────────────
    logger.info("=== STEP 5: Registry Push ===")
    pusher = HFHubPusher(repo_id=cfg["registry"]["hf_repo_id"], token=os.environ["HF_TOKEN"])
    pusher.create_repo()
    pusher.push_tokenizer(str(saved_tokenizer_path))
    if cfg["registry"]["push_shards"]:
        pusher.push_shards(cfg["artifacts"]["shards_dir"])

    # ── Step 6: Final Report ──────────────────────────
    logger.info("\n=== PIPELINE REPORT ===")
    logger.info(filter_pipeline.rejection_report())
    logger.info(f"Dedup rate: {deduplicator.dedup_rate:.2%}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")
    logger.info(f"Num shards: {stats['num_shards']}")