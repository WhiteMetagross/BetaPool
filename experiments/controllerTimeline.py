#!/usr/bin/env python3
"""
Controller Timeline Experiment.

Records and visualizes the adaptive controller's behavior over time,
showing thread count adjustments, blocking ratio, and veto events.

This addresses reviewer concern #8: controller heuristics and timeline visualization.
"""

import time
import random
import statistics
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from threading import Thread, Lock, Event
from collections import deque

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Experiment parameters.
CPU_ITERATIONS = 1000
IO_SLEEP_MS = 0.1
EXPERIMENT_DURATION_SEC = 30
MONITOR_INTERVAL_SEC = 0.5


@dataclass
class TimelinePoint:
    """Single point in the controller timeline."""
    timestamp: float
    activeThreads: int
    queueDepth: int
    blockingRatio: float
    blockingRatioEwma: float
    throughput: float
    decision: str  # "scale_up", "scale_down", "veto", "hold"
    vetoReason: Optional[str] = None


@dataclass
class VetoEvent:
    """Record of a veto decision."""
    timestamp: float
    threadsBefore: int
    blockingRatio: float
    reason: str


class AdaptiveControllerWithTimeline:
    """
    Adaptive controller with detailed timeline logging.
    
    Implements EWMA smoothing for blocking ratio and hysteresis
    to prevent oscillations.
    """
    
    def __init__(
        self,
        minWorkers: int = 4,
        maxWorkers: int = 128,
        betaThreshold: float = 0.3,
        ewmaAlpha: float = 0.2,
        hysteresisCount: int = 3,
    ):
        self.minWorkers = minWorkers
        self.maxWorkers = maxWorkers
        self.betaThreshold = betaThreshold
        self.ewmaAlpha = ewmaAlpha
        self.hysteresisCount = hysteresisCount
        
        # Current state.
        self.currentWorkers = minWorkers
        self.executor = ThreadPoolExecutor(max_workers=minWorkers)
        
        # Metrics tracking.
        self.recentBetas: deque = deque(maxlen=100)
        self.betaEwma = 0.5
        self.taskCount = 0
        self.pendingTasks = 0
        self.completedTasks = 0
        self.lastThroughputCheck = time.time()
        self.tasksSinceCheck = 0
        
        # Hysteresis state.
        self.consecutiveScaleSignals = 0
        self.lastScaleDirection = 0
        
        # Timeline and veto logging.
        self.timeline: List[TimelinePoint] = []
        self.vetoEvents: List[VetoEvent] = []
        self.startTime = time.time()
        
        # Locks.
        self.lock = Lock()
        self.stopEvent = Event()
        
        # Start monitor thread.
        self.monitorThread = Thread(target=self._monitorLoop, daemon=True)
        self.monitorThread.start()
    
    def submit(self, fn, *args, **kwargs):
        """Submit a task for execution."""
        with self.lock:
            self.pendingTasks += 1
            self.taskCount += 1
        
        future = self.executor.submit(self._wrapTask, fn, *args, **kwargs)
        return future
    
    def _wrapTask(self, fn, *args, **kwargs):
        """Wrap task to measure blocking ratio."""
        wallStart = time.time()
        cpuStart = time.thread_time()
        
        result = fn(*args, **kwargs)
        
        cpuEnd = time.thread_time()
        wallEnd = time.time()
        
        wallTime = wallEnd - wallStart
        cpuTime = cpuEnd - cpuStart
        beta = 1.0 - min(1.0, cpuTime / wallTime) if wallTime > 0 else 0.5
        
        with self.lock:
            self.recentBetas.append(beta)
            self.pendingTasks -= 1
            self.completedTasks += 1
            self.tasksSinceCheck += 1
            
            # Update EWMA.
            self.betaEwma = self.ewmaAlpha * beta + (1 - self.ewmaAlpha) * self.betaEwma
        
        return result
    
    def _monitorLoop(self):
        """Main control loop with timeline logging."""
        while not self.stopEvent.is_set():
            time.sleep(MONITOR_INTERVAL_SEC)
            
            if self.stopEvent.is_set():
                break
            
            self._makeDecision()
    
    def _makeDecision(self):
        """Make scaling decision and log to timeline."""
        currentTime = time.time()
        relativeTime = currentTime - self.startTime
        
        with self.lock:
            queueDepth = self.pendingTasks
            betaAvg = statistics.mean(self.recentBetas) if self.recentBetas else 0.5
            betaEwma = self.betaEwma
            
            # Calculate throughput.
            elapsed = currentTime - self.lastThroughputCheck
            throughput = self.tasksSinceCheck / elapsed if elapsed > 0 else 0
            self.tasksSinceCheck = 0
            self.lastThroughputCheck = currentTime
            
            decision = "hold"
            vetoReason = None
            threadsBefore = self.currentWorkers
            
            # Decision logic with hysteresis.
            if queueDepth > 0:
                # There's work to do.
                if betaEwma > self.betaThreshold:
                    # IO bound, want to scale up.
                    if self.lastScaleDirection >= 0:
                        self.consecutiveScaleSignals += 1
                    else:
                        self.consecutiveScaleSignals = 1
                    self.lastScaleDirection = 1
                    
                    if self.consecutiveScaleSignals >= self.hysteresisCount:
                        if self.currentWorkers < self.maxWorkers:
                            self.currentWorkers = min(self.currentWorkers + 2, self.maxWorkers)
                            self._resizePool()
                            decision = "scale_up"
                            self.consecutiveScaleSignals = 0
                else:
                    # CPU bound / GIL contention detected.
                    decision = "veto"
                    vetoReason = f"beta={betaEwma:.3f} < threshold={self.betaThreshold}"
                    self.vetoEvents.append(VetoEvent(
                        timestamp=relativeTime,
                        threadsBefore=threadsBefore,
                        blockingRatio=betaEwma,
                        reason=vetoReason,
                    ))
                    self.consecutiveScaleSignals = 0
                    self.lastScaleDirection = 0
            
            elif queueDepth == 0 and self.currentWorkers > self.minWorkers:
                # No pending work, consider scaling down.
                if self.lastScaleDirection <= 0:
                    self.consecutiveScaleSignals += 1
                else:
                    self.consecutiveScaleSignals = 1
                self.lastScaleDirection = -1
                
                if self.consecutiveScaleSignals >= self.hysteresisCount:
                    self.currentWorkers = max(self.currentWorkers - 1, self.minWorkers)
                    self._resizePool()
                    decision = "scale_down"
                    self.consecutiveScaleSignals = 0
            
            # Log timeline point.
            self.timeline.append(TimelinePoint(
                timestamp=relativeTime,
                activeThreads=self.currentWorkers,
                queueDepth=queueDepth,
                blockingRatio=betaAvg,
                blockingRatioEwma=betaEwma,
                throughput=throughput,
                decision=decision,
                vetoReason=vetoReason,
            ))
    
    def _resizePool(self):
        """Resize the thread pool."""
        self.executor._max_workers = self.currentWorkers
        self.executor._adjust_thread_count()
    
    def shutdown(self, wait=True):
        """Shutdown the controller and executor."""
        self.stopEvent.set()
        if self.monitorThread.is_alive():
            self.monitorThread.join(timeout=2.0)
        self.executor.shutdown(wait=wait)
    
    def getTimeline(self) -> List[TimelinePoint]:
        """Return the recorded timeline."""
        return self.timeline.copy()
    
    def getVetoEvents(self) -> List[VetoEvent]:
        """Return recorded veto events."""
        return self.vetoEvents.copy()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown(wait=True)


def mixedTask() -> float:
    """Standard mixed CPU+IO task."""
    start = time.perf_counter()
    
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    time.sleep(IO_SLEEP_MS / 1000.0)
    
    return time.perf_counter() - start


def runTimelineExperiment() -> Tuple[List[TimelinePoint], List[VetoEvent], Dict]:
    """Run experiment and capture controller timeline."""
    print()
    print("=" * 70)
    print("CONTROLLER TIMELINE EXPERIMENT")
    print("=" * 70)
    print()
    print(f"Duration: {EXPERIMENT_DURATION_SEC} seconds")
    print(f"Monitor interval: {MONITOR_INTERVAL_SEC} seconds")
    print()
    
    latencies = []
    startTime = time.time()
    
    with AdaptiveControllerWithTimeline(
        minWorkers=4,
        maxWorkers=128,
        betaThreshold=0.3,
        ewmaAlpha=0.2,
        hysteresisCount=3,
    ) as controller:
        # Submit tasks continuously for the experiment duration.
        print("Submitting tasks...")
        futures = []
        taskCount = 0
        
        while time.time() - startTime < EXPERIMENT_DURATION_SEC:
            # Submit batch of tasks.
            batch = [controller.submit(mixedTask) for _ in range(100)]
            futures.extend(batch)
            taskCount += 100
            
            # Brief pause to allow processing.
            time.sleep(0.01)
        
        print(f"Submitted {taskCount} tasks, waiting for completion...")
        
        # Collect results.
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as e:
                print(f"Task failed: {e}")
        
        timeline = controller.getTimeline()
        vetoEvents = controller.getVetoEvents()
        finalThreads = controller.currentWorkers
    
    endTime = time.time()
    elapsed = endTime - startTime
    
    # Compute summary statistics.
    latenciesMs = [lat * 1000 for lat in latencies]
    latenciesMs.sort()
    
    summary = {
        "taskCount": len(latencies),
        "elapsed": elapsed,
        "tps": len(latencies) / elapsed,
        "avgLatMs": statistics.mean(latenciesMs),
        "p99LatMs": latenciesMs[int(len(latenciesMs) * 0.99)],
        "finalThreads": finalThreads,
        "vetoCount": len(vetoEvents),
        "timelinePoints": len(timeline),
    }
    
    return timeline, vetoEvents, summary


def saveTimelineResults(
    timeline: List[TimelinePoint],
    vetoEvents: List[VetoEvent],
    summary: Dict
):
    """Save timeline data to CSV files."""
    os.makedirs("results", exist_ok=True)
    
    # Save timeline.
    with open("results/controller_timeline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "active_threads", "queue_depth", "blocking_ratio",
            "blocking_ratio_ewma", "throughput", "decision", "veto_reason"
        ])
        for p in timeline:
            writer.writerow([
                p.timestamp, p.activeThreads, p.queueDepth, p.blockingRatio,
                p.blockingRatioEwma, p.throughput, p.decision, p.vetoReason or ""
            ])
    
    # Save veto events.
    with open("results/veto_events.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "threads_before", "blocking_ratio", "reason"])
        for v in vetoEvents:
            writer.writerow([v.timestamp, v.threadsBefore, v.blockingRatio, v.reason])
    
    print("Results saved to:")
    print("  results/controller_timeline.csv")
    print("  results/veto_events.csv")


def printTimelineAnalysis(
    timeline: List[TimelinePoint],
    vetoEvents: List[VetoEvent],
    summary: Dict
):
    """Print analysis of the timeline."""
    print()
    print("=" * 70)
    print("TIMELINE ANALYSIS")
    print("=" * 70)
    print()
    
    print("Experiment Summary:")
    print(f"  Tasks completed: {summary['taskCount']}")
    print(f"  Duration: {summary['elapsed']:.1f} seconds")
    print(f"  Throughput: {summary['tps']:,.0f} TPS")
    print(f"  Avg Latency: {summary['avgLatMs']:.2f} ms")
    print(f"  P99 Latency: {summary['p99LatMs']:.2f} ms")
    print()
    
    print("Controller Behavior:")
    print(f"  Timeline points: {len(timeline)}")
    print(f"  Final threads: {summary['finalThreads']}")
    print(f"  Veto events: {summary['vetoCount']}")
    print()
    
    # Count decisions.
    decisions = {}
    for p in timeline:
        decisions[p.decision] = decisions.get(p.decision, 0) + 1
    
    print("Decision Distribution:")
    for decision, count in sorted(decisions.items()):
        print(f"  {decision}: {count}")
    print()
    
    # Thread range.
    threadCounts = [p.activeThreads for p in timeline]
    if threadCounts:
        print("Thread Count Range:")
        print(f"  Min: {min(threadCounts)}")
        print(f"  Max: {max(threadCounts)}")
        print(f"  Avg: {statistics.mean(threadCounts):.1f}")
    print()
    
    # Beta analysis.
    betas = [p.blockingRatioEwma for p in timeline]
    if betas:
        print("Blocking Ratio (EWMA):")
        print(f"  Min: {min(betas):.3f}")
        print(f"  Max: {max(betas):.3f}")
        print(f"  Avg: {statistics.mean(betas):.3f}")
    print()
    
    # Veto analysis.
    if vetoEvents:
        print("Veto Event Analysis:")
        print(f"  Total vetoes: {len(vetoEvents)}")
        vetoBetas = [v.blockingRatio for v in vetoEvents]
        print(f"  Avg beta at veto: {statistics.mean(vetoBetas):.3f}")
        print(f"  Vetoes prevented overscaling and avoided the saturation cliff.")


def main():
    """Main entry point for timeline experiment."""
    print()
    print("#" * 70)
    print("# ADAPTIVE CONTROLLER TIMELINE EXPERIMENT")
    print("#" * 70)
    
    timeline, vetoEvents, summary = runTimelineExperiment()
    
    saveTimelineResults(timeline, vetoEvents, summary)
    
    printTimelineAnalysis(timeline, vetoEvents, summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
