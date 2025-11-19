"""Script to view telemetry data and agent metrics."""

import json
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

from utils.telemetry import get_telemetry_collector
from utils.config import LOGS_DIR


def print_agent_stats(stats: dict):
    """Print formatted agent statistics."""
    print("\n" + "="*80)
    print("AGENT STATISTICS")
    print("="*80)
    
    for agent_name, agent_data in stats.items():
        print(f"\n🤖 {agent_name}")
        print(f"   Total Operations: {agent_data['total_operations']}")
        print(f"   Success: {agent_data['success_count']} | Errors: {agent_data['error_count']}")
        print(f"   Avg Duration: {agent_data['avg_duration_ms']:.2f}ms")
        print(f"   Total Duration: {agent_data['total_duration_ms']:.2f}ms")
        
        # Operation breakdown
        op_counts = defaultdict(int)
        for op in agent_data['operations']:
            op_counts[op] += 1
        if op_counts:
            print(f"   Operations:")
            for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {op}: {count}")


def print_recent_metrics(metrics: list, limit: int = 20):
    """Print recent metrics."""
    print("\n" + "="*80)
    print(f"RECENT METRICS (Last {min(limit, len(metrics))})")
    print("="*80)
    
    for metric in metrics[-limit:]:
        status = "❌" if metric.get("error") else "✅"
        print(f"\n{status} {metric['agent_name']} | {metric['operation']}")
        print(f"   Time: {metric['timestamp']}")
        print(f"   Duration: {metric.get('duration_ms', 0):.2f}ms")
        if metric.get("error"):
            print(f"   Error: {metric['error']}")
        if metric.get("metadata"):
            print(f"   Metadata: {metric['metadata']}")


def view_telemetry_log(log_file: Path, limit: int = 50):
    """View telemetry from JSONL log file."""
    if not log_file.exists():
        print(f"Telemetry log file not found: {log_file}")
        return
    
    print(f"\nReading telemetry from: {log_file}")
    print("="*80)
    
    metrics = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                metric = json.loads(line.strip())
                metrics.append(metric)
            except json.JSONDecodeError:
                continue
    
    if not metrics:
        print("No metrics found in log file.")
        return
    
    print(f"\nTotal metrics in log: {len(metrics)}")
    
    # Group by agent
    agent_stats = defaultdict(lambda: {
        "total_operations": 0,
        "total_duration_ms": 0.0,
        "success_count": 0,
        "error_count": 0,
        "operations": []
    })
    
    for metric in metrics:
        agent = metric["agent_name"]
        agent_stats[agent]["total_operations"] += 1
        if metric.get("duration_ms"):
            agent_stats[agent]["total_duration_ms"] += metric["duration_ms"]
        if metric.get("error"):
            agent_stats[agent]["error_count"] += 1
        else:
            agent_stats[agent]["success_count"] += 1
        agent_stats[agent]["operations"].append(metric["operation"])
    
    # Calculate averages
    for agent_data in agent_stats.values():
        if agent_data["total_operations"] > 0:
            agent_data["avg_duration_ms"] = (
                agent_data["total_duration_ms"] / agent_data["total_operations"]
            )
    
    print_agent_stats(dict(agent_stats))
    print_recent_metrics(metrics, limit)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View telemetry data and agent metrics")
    parser.add_argument(
        "--live",
        action="store_true",
        help="View live metrics from current session"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOGS_DIR / "telemetry.jsonl",
        help="Path to telemetry log file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent metrics to display"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export metrics to JSON file"
    )
    
    args = parser.parse_args()
    
    if args.live:
        # View live metrics
        collector = get_telemetry_collector()
        stats = collector.get_agent_stats()
        recent = collector.get_recent_metrics(limit=args.limit)
        
        print_agent_stats(stats)
        print_recent_metrics(recent, limit=args.limit)
        
        if args.export:
            export_path = collector.export_metrics(args.export)
            print(f"\n✅ Metrics exported to: {export_path}")
    else:
        # View from log file
        view_telemetry_log(args.log_file, limit=args.limit)
        
        if args.export:
            # Read and export
            metrics = []
            with open(args.log_file, "r") as f:
                for line in f:
                    try:
                        metrics.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            export_data = {
                "total_metrics": len(metrics),
                "metrics": metrics
            }
            
            with open(args.export, "w") as f:
                json.dump(export_data, f, indent=2)
            
            print(f"\n✅ Metrics exported to: {args.export}")


if __name__ == "__main__":
    main()

