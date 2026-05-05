import os
import sys
import json
import asyncio
import argparse
import pandas as pd
import aiohttp

sys.path.append(os.path.dirname(__file__))
from optimized_network_feature_collector import NetworkFeatureCollector


async def enrich_dataset(input_path: str, output_path: str, batch_size: int = 200):
    df = pd.read_csv(input_path)
    if "url" not in df.columns:
        raise ValueError("Dataset missing 'url' column")

    if "network_features" not in df.columns:
        df["network_features"] = None

    collector = NetworkFeatureCollector()
    await collector.load_cache()
    collector.init_redis_client()

    url_series = df["url"].fillna("").astype(str).str.strip()
    needs_collect = []
    for idx, url in url_series.items():
        if not url:
            continue
        existing = df.at[idx, "network_features"]
        if pd.isna(existing) or str(existing).strip() == "":
            needs_collect.append(url)

    unique_urls = list(dict.fromkeys(needs_collect))
    total_urls = len(unique_urls)
    print(f"Need collect unique URLs: {total_urls}")

    if total_urls > 0:
        connector = aiohttp.TCPConnector(ssl=False, limit=batch_size, limit_per_host=10, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_connect=2, sock_read=3)
        features_by_url = {}

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for start in range(0, total_urls, batch_size):
                end = min(start + batch_size, total_urls)
                batch_urls = unique_urls[start:end]
                tasks = [collector.async_collect_network_features(session, u) for u in batch_urls]
                batch_features = await asyncio.gather(*tasks)
                for u, f in zip(batch_urls, batch_features):
                    features_by_url[u] = ",".join([f"{x:.4f}" for x in f[:8]])
                print(f"Collected {end}/{total_urls}")

        for idx, url in url_series.items():
            if url in features_by_url:
                df.at[idx, "network_features"] = features_by_url[url]

    await collector.save_cache()
    if collector.redis_client:
        try:
            collector.redis_client.close()
        except Exception:
            pass

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    non_empty = df["network_features"].fillna("").astype(str).str.strip().ne("").sum()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}, network_features non-empty: {int(non_empty)} ({non_empty/len(df):.2%})")


def main():
    parser = argparse.ArgumentParser(description="Enrich dataset network_features without dropping rows")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=200, help="Async batch size")
    args = parser.parse_args()

    asyncio.run(enrich_dataset(args.input, args.output, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
