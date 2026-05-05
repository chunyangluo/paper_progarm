#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nightly network feature collection pipeline for large datasets.

Capabilities:
- Chunked collection on large CSV datasets
- Resume from previous progress automatically
- Run limited chunks per execution (good for night jobs)
- Merge collected chunks back into a full dataset
- Coverage report after each run
"""

import os
import sys
import json
import argparse
import asyncio
from datetime import datetime

import aiohttp
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from optimized_network_feature_collector import NetworkFeatureCollector


def _safe_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)


class NightlyCollector:
    def __init__(
        self,
        input_path: str,
        run_name: str = "",
        chunk_size: int = 5000,
        batch_size: int = 200,
        chunks_per_run: int = 2,
        output_root: str = "",
    ):
        self.input_path = input_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.chunks_per_run = chunks_per_run

        if output_root:
            root = output_root
        else:
            root = os.path.join(os.path.dirname(os.path.dirname(input_path)), "nightly_network_runs")

        if not run_name:
            run_name = _safe_name(input_path)

        self.run_dir = os.path.join(root, run_name)
        self.chunks_dir = os.path.join(self.run_dir, "chunks")
        self.state_path = os.path.join(self.run_dir, "state.json")
        self.report_path = os.path.join(self.run_dir, "latest_report.json")
        self.merged_output_path = os.path.join(
            self.run_dir,
            f"{_safe_name(input_path)}_with_network_features_merged.csv",
        )

        os.makedirs(self.chunks_dir, exist_ok=True)
        self.collector = NetworkFeatureCollector()

    def _write_report(self, data: dict):
        payload = dict(data)
        payload["updated_at"] = datetime.now().isoformat()
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_state(self, total_rows: int) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Keep compatibility if dataset changed.
            if state.get("total_rows") != total_rows:
                state = self._new_state(total_rows)
        else:
            state = self._new_state(total_rows)
        return state

    def _new_state(self, total_rows: int) -> dict:
        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        return {
            "input_path": self.input_path,
            "chunk_size": self.chunk_size,
            "total_rows": total_rows,
            "total_chunks": total_chunks,
            "next_chunk_idx": 0,
            "completed_chunks": [],
            "updated_at": datetime.now().isoformat(),
        }

    def _save_state(self, state: dict):
        state["updated_at"] = datetime.now().isoformat()
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _count_non_empty_from_chunks(self) -> int:
        if not os.path.exists(self.chunks_dir):
            return 0
        total = 0
        for name in os.listdir(self.chunks_dir):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(self.chunks_dir, name)
            try:
                cdf = pd.read_csv(path, usecols=["network_features"])
                total += int(cdf["network_features"].fillna("").astype(str).str.strip().ne("").sum())
            except Exception:
                continue
        return total

    async def _collect_chunk(
        self,
        df: pd.DataFrame,
        chunk_idx: int,
        total_chunks: int,
        completed_count: int,
        global_stats: dict,
    ) -> dict:
        chunk_file = os.path.join(self.chunks_dir, f"chunk_{chunk_idx:06d}.csv")
        if os.path.exists(chunk_file):
            return {"chunk_file": chunk_file, "processed_urls": 0, "assigned_rows": 0}

        start = chunk_idx * self.chunk_size
        end = min((chunk_idx + 1) * self.chunk_size, len(df))
        chunk = df.iloc[start:end].copy()
        chunk["__row_id__"] = list(range(start, end))

        # Determine rows needing collection.
        has_nf_col = "network_features" in chunk.columns
        if not has_nf_col:
            chunk["network_features"] = None

        url_series = chunk["url"].fillna("").astype(str).str.strip() if "url" in chunk.columns else pd.Series([""] * len(chunk))
        needs_collect = []
        url_row_counts = {}
        for idx, url in zip(chunk["__row_id__"], url_series):
            if not url:
                continue
            existing = chunk.loc[chunk["__row_id__"] == idx, "network_features"].values[0]
            if pd.isna(existing) or str(existing).strip() == "":
                needs_collect.append(url)
                url_row_counts[url] = url_row_counts.get(url, 0) + 1

        unique_urls = list(dict.fromkeys(needs_collect))
        features_by_url = {}
        current_total = len(needs_collect)
        processed = 0

        if unique_urls:
            connector = aiohttp.TCPConnector(
                ssl=False,
                limit=self.batch_size,
                limit_per_host=10,
                ttl_dns_cache=300,
            )
            timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_connect=2, sock_read=3)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                for pos in range(0, len(unique_urls), self.batch_size):
                    batch = unique_urls[pos:pos + self.batch_size]
                    tasks = [self.collector.async_collect_network_features(session, u) for u in batch]
                    results = await asyncio.gather(*tasks)
                    for u, feat in zip(batch, results):
                        features_by_url[u] = ",".join(f"{x:.4f}" for x in feat[:8])
                    processed_rows = sum(url_row_counts.get(u, 0) for u in batch)
                    processed += processed_rows
                    global_stats["processed_urls"] += processed_rows
                    self._write_report({
                        "input_path": self.input_path,
                        "run_dir": self.run_dir,
                        "total_chunks": total_chunks,
                        "completed_chunks": completed_count,
                        "remaining_chunks": total_chunks - completed_count,
                        "done": False,
                        "running": True,
                        "phase": "collecting",
                        "current_chunk_idx": chunk_idx,
                        "current_chunk_total_urls": current_total,
                        "current_chunk_processed_urls": processed,
                        "current_chunk_progress": (processed / current_total) if current_total else 1.0,
                        "global_progress": (completed_count + ((processed / current_total) if current_total else 1.0)) / total_chunks,
                        "global_target_urls": global_stats["target_urls"],
                        "global_processed_urls": global_stats["processed_urls"],
                        "global_processed_progress": (
                            global_stats["processed_urls"] / global_stats["target_urls"]
                            if global_stats["target_urls"] else 1.0
                        ),
                        "global_non_empty_network_features": global_stats["non_empty_network_features"],
                        "global_coverage_progress": (
                            global_stats["non_empty_network_features"] / global_stats["total_rows"]
                            if global_stats["total_rows"] else 0.0
                        ),
                    })

        # Save only row_id + collected/final network_features for this chunk.
        out = chunk[["__row_id__", "network_features"]].copy()
        if features_by_url:
            assigned_rows = 0
            for i, url in zip(chunk["__row_id__"], url_series):
                if url in features_by_url:
                    out.loc[out["__row_id__"] == i, "network_features"] = features_by_url[url]
                    assigned_rows += 1
            global_stats["non_empty_network_features"] += assigned_rows
        else:
            assigned_rows = 0

        out.to_csv(chunk_file, index=False, encoding="utf-8-sig")
        return {"chunk_file": chunk_file, "processed_urls": processed, "assigned_rows": assigned_rows}

    async def run_collect(self, merge_when_done: bool = True):
        df = pd.read_csv(self.input_path)
        if "url" not in df.columns:
            raise ValueError("Input dataset missing 'url' column")

        await self.collector.load_cache()
        self.collector.init_redis_client()

        state = self._load_state(len(df))
        total_chunks = state["total_chunks"]
        completed = set(state["completed_chunks"])
        total_rows = len(df)
        existing_non_empty = self._count_non_empty_from_chunks()
        if "url" in df.columns:
            url_non_empty = df["url"].fillna("").astype(str).str.strip().ne("")
        else:
            url_non_empty = pd.Series([False] * total_rows)
        target_urls = int(url_non_empty.sum())
        global_stats = {
            "total_rows": total_rows,
            "target_urls": target_urls,
            "processed_urls": int(existing_non_empty),
            "non_empty_network_features": int(existing_non_empty),
        }

        # Build candidate chunk queue from next index then wrap.
        queue = [i for i in range(state["next_chunk_idx"], total_chunks)] + [i for i in range(0, state["next_chunk_idx"])]
        queue = [i for i in queue if i not in completed]
        to_run = queue[: self.chunks_per_run]

        self._write_report({
            "input_path": self.input_path,
            "run_dir": self.run_dir,
            "total_chunks": total_chunks,
            "completed_chunks": len(completed),
            "remaining_chunks": total_chunks - len(completed),
            "done": False,
            "running": True,
            "phase": "collecting",
            "current_chunk_idx": None,
            "current_chunk_total_urls": 0,
            "current_chunk_processed_urls": 0,
            "current_chunk_progress": 0.0,
            "global_progress": (len(completed) / total_chunks) if total_chunks else 0.0,
            "global_target_urls": global_stats["target_urls"],
            "global_processed_urls": global_stats["processed_urls"],
            "global_processed_progress": 0.0,
            "global_non_empty_network_features": global_stats["non_empty_network_features"],
            "global_coverage_progress": (
                global_stats["non_empty_network_features"] / global_stats["total_rows"]
                if global_stats["total_rows"] else 0.0
            ),
        })

        try:
            if not to_run:
                print("No remaining chunks to collect. All chunks completed.")
            else:
                print(f"Running chunks this execution: {to_run}")
                for cidx in to_run:
                    await self._collect_chunk(df, cidx, total_chunks, len(completed), global_stats)
                    completed.add(cidx)
                    state["completed_chunks"] = sorted(completed)
                    state["next_chunk_idx"] = (cidx + 1) % total_chunks
                    self._save_state(state)
                    self._write_report({
                        "input_path": self.input_path,
                        "run_dir": self.run_dir,
                        "total_chunks": total_chunks,
                        "completed_chunks": len(completed),
                        "remaining_chunks": total_chunks - len(completed),
                        "done": False,
                        "running": True,
                        "phase": "collecting",
                        "current_chunk_idx": None,
                        "current_chunk_total_urls": 0,
                        "current_chunk_processed_urls": 0,
                        "current_chunk_progress": 0.0,
                        "global_progress": (len(completed) / total_chunks) if total_chunks else 1.0,
                        "global_target_urls": global_stats["target_urls"],
                        "global_processed_urls": global_stats["processed_urls"],
                        "global_processed_progress": (
                            global_stats["processed_urls"] / global_stats["target_urls"]
                            if global_stats["target_urls"] else 1.0
                        ),
                        "global_non_empty_network_features": global_stats["non_empty_network_features"],
                        "global_coverage_progress": (
                            global_stats["non_empty_network_features"] / global_stats["total_rows"]
                            if global_stats["total_rows"] else 0.0
                        ),
                    })
                    print(f"Chunk {cidx} completed.")
        except Exception as e:
            self._write_report({
                "input_path": self.input_path,
                "run_dir": self.run_dir,
                "total_chunks": total_chunks,
                "completed_chunks": len(completed),
                "remaining_chunks": total_chunks - len(completed),
                "done": False,
                "running": False,
                "phase": "failed",
                "error": str(e),
                "current_chunk_idx": None,
                "current_chunk_total_urls": 0,
                "current_chunk_processed_urls": 0,
                "current_chunk_progress": 0.0,
                "global_progress": (len(completed) / total_chunks) if total_chunks else 0.0,
                "global_target_urls": global_stats["target_urls"],
                "global_processed_urls": global_stats["processed_urls"],
                "global_processed_progress": (
                    global_stats["processed_urls"] / global_stats["target_urls"]
                    if global_stats["target_urls"] else 0.0
                ),
                "global_non_empty_network_features": global_stats["non_empty_network_features"],
                "global_coverage_progress": (
                    global_stats["non_empty_network_features"] / global_stats["total_rows"]
                    if global_stats["total_rows"] else 0.0
                ),
            })
            raise
        finally:
            await self.collector.save_cache()
            if self.collector.redis_client:
                try:
                    self.collector.redis_client.close()
                except Exception:
                    pass

        done = len(completed) == total_chunks
        report = {
            "input_path": self.input_path,
            "run_dir": self.run_dir,
            "total_chunks": total_chunks,
            "completed_chunks": len(completed),
            "remaining_chunks": total_chunks - len(completed),
            "done": done,
            "running": False,
            "phase": "completed" if done else "idle",
            "current_chunk_idx": None,
            "current_chunk_total_urls": 0,
            "current_chunk_processed_urls": 0,
            "current_chunk_progress": 0.0,
            "global_progress": (len(completed) / total_chunks) if total_chunks else 1.0,
            "global_target_urls": global_stats["target_urls"],
            "global_processed_urls": global_stats["processed_urls"],
            "global_processed_progress": (
                global_stats["processed_urls"] / global_stats["target_urls"]
                if global_stats["target_urls"] else 1.0
            ),
            "global_non_empty_network_features": global_stats["non_empty_network_features"],
            "global_coverage_progress": (
                global_stats["non_empty_network_features"] / global_stats["total_rows"]
                if global_stats["total_rows"] else 0.0
            ),
        }
        self._write_report(report)

        if done and merge_when_done:
            self.run_merge()
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))

    def run_merge(self):
        df = pd.read_csv(self.input_path)
        if "network_features" not in df.columns:
            df["network_features"] = None

        chunk_files = sorted(
            [
                os.path.join(self.chunks_dir, f)
                for f in os.listdir(self.chunks_dir)
                if f.endswith(".csv")
            ]
        )
        if not chunk_files:
            raise RuntimeError("No chunk files found; nothing to merge.")

        merged_rows = 0
        for fp in chunk_files:
            cdf = pd.read_csv(fp)
            if "__row_id__" not in cdf.columns or "network_features" not in cdf.columns:
                continue
            for row_id, nf in zip(cdf["__row_id__"], cdf["network_features"]):
                if pd.isna(nf) or str(nf).strip() == "":
                    continue
                df.at[int(row_id), "network_features"] = nf
                merged_rows += 1

        df.to_csv(self.merged_output_path, index=False, encoding="utf-8-sig")

        non_empty = df["network_features"].fillna("").astype(str).str.strip().ne("").sum()
        coverage = non_empty / len(df) if len(df) else 0.0
        result = {
            "merged_output_path": self.merged_output_path,
            "rows": len(df),
            "non_empty_network_features": int(non_empty),
            "coverage": coverage,
            "chunk_files": len(chunk_files),
            "merged_assignments": int(merged_rows),
            "running": False,
            "phase": "merged",
        }
        self._write_report(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Nightly chunked network feature collection pipeline")
    parser.add_argument("--input", required=True, help="Input dataset csv path")
    parser.add_argument("--run-name", default="", help="Run name folder (default: derived from input name)")
    parser.add_argument("--output-root", default="", help="Root dir for run artifacts")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Rows per chunk")
    parser.add_argument("--batch-size", type=int, default=200, help="Async URL batch size")
    parser.add_argument("--chunks-per-run", type=int, default=2, help="How many chunks to process this run")
    parser.add_argument("--merge-only", action="store_true", help="Skip collecting and only merge chunk outputs")
    parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto merge when all chunks completed")
    args = parser.parse_args()

    collector = NightlyCollector(
        input_path=args.input,
        run_name=args.run_name,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        chunks_per_run=args.chunks_per_run,
        output_root=args.output_root,
    )

    if args.merge_only:
        collector.run_merge()
    else:
        asyncio.run(collector.run_collect(merge_when_done=not args.no_auto_merge))


if __name__ == "__main__":
    main()
