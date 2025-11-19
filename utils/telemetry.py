"""Telemetry and observability for agent operations."""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path
from loguru import logger

from utils.config import LOGS_DIR


@dataclass
class AgentMetric:
    """Single metric for an agent operation."""
    agent_name: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None, 
                 error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Mark metric as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if output_data:
            self.output_data = output_data
        if error:
            self.error = error
        if metadata:
            self.metadata.update(metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_data": self._serialize_data(self.input_data),
            "output_data": self._serialize_data(self.output_data),
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat()
        }
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON compatibility."""
        if data is None:
            return None
        try:
            # Try to serialize as-is
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # If it contains non-serializable objects, convert to string
            return str(data)


class TelemetryCollector:
    """Collects and stores telemetry data for agents."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize telemetry collector.
        
        Args:
            log_dir: Directory to store telemetry logs (defaults to LOGS_DIR)
        """
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[AgentMetric] = []
        self.active_metrics: Dict[str, AgentMetric] = {}
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_operations": 0,
            "total_duration_ms": 0.0,
            "success_count": 0,
            "error_count": 0,
            "avg_duration_ms": 0.0,
            "operations": []
        })
        
        # Create telemetry log file
        self.telemetry_log = self.log_dir / "telemetry.jsonl"
    
    def start_operation(self, agent_name: str, operation: str, 
                       input_data: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an operation.
        
        Args:
            agent_name: Name of the agent
            operation: Operation name
            input_data: Input data for the operation
            metadata: Additional metadata
        
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{agent_name}_{operation}_{int(time.time() * 1000000)}"
        
        metric = AgentMetric(
            agent_name=agent_name,
            operation=operation,
            start_time=time.time(),
            input_data=input_data or {},
            metadata=metadata or {}
        )
        
        self.active_metrics[operation_id] = metric
        
        # Log start
        logger.info(
            f"🔍 [TELEMETRY] Agent: {agent_name} | Operation: {operation} | Started",
            extra={"telemetry": True, "agent": agent_name, "operation": operation}
        )
        
        return operation_id
    
    def end_operation(self, operation_id: str, 
                     output_data: Optional[Dict[str, Any]] = None,
                     error: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        End tracking an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            output_data: Output data from the operation
            error: Error message if operation failed
            metadata: Additional metadata
        """
        if operation_id not in self.active_metrics:
            logger.warning(f"Operation {operation_id} not found in active metrics")
            return
        
        metric = self.active_metrics.pop(operation_id)
        metric.complete(output_data, error, metadata)
        
        # Add to metrics list
        self.metrics.append(metric)
        
        # Update agent stats
        stats = self.agent_stats[metric.agent_name]
        stats["total_operations"] += 1
        stats["total_duration_ms"] += metric.duration_ms or 0
        if error:
            stats["error_count"] += 1
        else:
            stats["success_count"] += 1
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_operations"]
        stats["operations"].append(metric.operation)
        
        # Log completion
        status = "❌ ERROR" if error else "✅ SUCCESS"
        logger.info(
            f"🔍 [TELEMETRY] Agent: {metric.agent_name} | Operation: {metric.operation} | "
            f"{status} | Duration: {metric.duration_ms:.2f}ms",
            extra={
                "telemetry": True,
                "agent": metric.agent_name,
                "operation": metric.operation,
                "duration_ms": metric.duration_ms,
                "error": error
            }
        )
        
        # Write to telemetry log file (JSONL format)
        self._write_metric(metric)
    
    def _write_metric(self, metric: AgentMetric):
        """Write metric to telemetry log file."""
        try:
            with open(self.telemetry_log, "a") as f:
                f.write(json.dumps(metric.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write telemetry metric: {e}")
    
    def get_agent_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for an agent or all agents.
        
        Args:
            agent_name: Specific agent name, or None for all agents
        
        Returns:
            Statistics dictionary
        """
        if agent_name:
            return self.agent_stats.get(agent_name, {})
        return dict(self.agent_stats)
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics.
        
        Args:
            limit: Maximum number of metrics to return
        
        Returns:
            List of metric dictionaries
        """
        recent = self.metrics[-limit:] if len(self.metrics) > limit else self.metrics
        return [m.to_dict() for m in recent]
    
    def export_metrics(self, file_path: Optional[Path] = None) -> Path:
        """
        Export all metrics to a JSON file.
        
        Args:
            file_path: Output file path (defaults to telemetry_export.json in log dir)
        
        Returns:
            Path to exported file
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.log_dir / f"telemetry_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics),
            "agent_stats": dict(self.agent_stats),
            "metrics": [m.to_dict() for m in self.metrics]
        }
        
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {file_path}")
        return file_path
    
    def clear_metrics(self):
        """Clear all collected metrics (useful for testing)."""
        self.metrics.clear()
        self.active_metrics.clear()
        self.agent_stats.clear()


# Global telemetry collector instance
_telemetry_collector: Optional[TelemetryCollector] = None


def get_telemetry_collector() -> TelemetryCollector:
    """Get or create global telemetry collector."""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def track_agent_operation(agent_name: str, operation: str):
    """
    Decorator to track agent operations.
    
    Usage:
        @track_agent_operation("QueryValidationAgent", "validate")
        def validate(self, query: str):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_telemetry_collector()
            
            # Extract input data (limit size for logging)
            input_data = {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            # Add first arg if it's a string (likely the query)
            if args and isinstance(args[0], str) and len(args[0]) < 500:
                input_data["first_arg"] = args[0]
            
            operation_id = collector.start_operation(
                agent_name=agent_name,
                operation=operation,
                input_data=input_data,
                metadata={"function": func.__name__}
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Extract output data (limit size)
                output_data = {}
                if isinstance(result, dict):
                    # Include keys and sample values
                    output_data = {
                        "keys": list(result.keys()),
                        "has_data": True
                    }
                    # Include small values
                    for k, v in result.items():
                        if isinstance(v, (str, int, float, bool)) and len(str(v)) < 200:
                            output_data[k] = v
                
                collector.end_operation(
                    operation_id,
                    output_data=output_data,
                    metadata={"result_type": type(result).__name__}
                )
                
                return result
            except Exception as e:
                collector.end_operation(
                    operation_id,
                    error=str(e),
                    metadata={"exception_type": type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator

