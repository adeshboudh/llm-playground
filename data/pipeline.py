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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def _clean_stream(downloader, filter_pipeline, deduplicator, dedup_enabled=True):
    passed = 0
    for i, text in enumerate(downloader.iterate()):
        if i % 100_000 == 0:
            logger.info(f"  scanned {i:,} docs | clean so far: {passed:,}")
        if not filter_pipeline.apply(text):
            continue
        if dedup_enabled and deduplicator.is_duplicate(text, doc_id=str(i)):
            continue
        passed += 1
        yield text

def run_pipeline(config_path: str, token: str = os.environ["HF_TOKEN"]):
    token = token or os.environ["HF_TOKEN"]

    # ── Load Config ──────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ic = cfg["ingestion"]
    cc = cfg["cleaning"]
    tc = cfg["tokenizer"]
    ac = cfg["artifacts"]

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
    tokenizer_path = ac["tokenizer_path"]
    if os.path.exists(tokenizer_path):
        logger.info(f"=== STEP 3: Tokenizer found at {tokenizer_path} — skipping training ===")
        trainer = BPETokenizerTrainer.load(tokenizer_path)
    else:
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Train it first with: python -m data.tokenizer.trainer"
        )

    # ── Step 4: Stream → encode shards ───────────────────────────────────
    logger.info("=== STEP 4: Streaming Encode → Shards ===")
    logger.info(f"  target: 50 shards × 100M tokens = 5B tokens")
    os.makedirs(ac["shards_dir"], exist_ok=True)

    encoder = ShardEncoder(
        tokenizer=trainer,
        output_dir=ac["shards_dir"],
        shard_size=tc["shard_size_tokens"],
    )
    stats = encoder.encode_stream(
        _clean_stream(
            downloader, filter_pipeline, deduplicator,
            dedup_enabled=cc["dedup_enabled"]
        )
    )
    logger.info(f"Encoding complete: {stats}")


    # ── Step 5: Push to HF Hub ────────────────────────────────────────────
    logger.info("=== STEP 5: Registry Push ===")
    pusher = HFHubPusher(repo_id=cfg["registry"]["hf_repo_id"], token=token)
    pusher.create_repo()
    pusher.push_tokenizer(tokenizer_path)
    if cfg["registry"]["push_shards"]:
        pusher.push_shards(ac["shards_dir"])

    # ── Step 6: Report ────────────────────────────────────────────────────
    logger.info("=== PIPELINE REPORT ===")
    logger.info(filter_pipeline.rejection_report())
    logger.info(f"Dedup rate       : {deduplicator.dedup_rate:.2%}")
    logger.info(f"Total tokens     : {stats['total_tokens']:,}")
    logger.info(f"Shards written   : {stats['num_shards']}")

if __name__ == "__main__":
    run_pipeline("data/configs/data_config.yaml")