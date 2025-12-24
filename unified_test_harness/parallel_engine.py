"""
Parallel Generation Engine for VectorReVamp

High-performance, resource-aware parallel test generation with intelligent
load balancing and fault tolerance. Inspired by vector_revamp's scalable
data generation architecture.
"""

import os
import time
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import queue
import signal
import resource

logger = logging.getLogger(__name__)


@dataclass
class ResourceProfile:
    """System resource profile for optimization decisions."""
    cpu_count: int
    cpu_percent: float
    memory_total: int  # bytes
    memory_available: int  # bytes
    memory_percent: float
    disk_free: int  # bytes
    load_average: Tuple[float, float, float]

    @classmethod
    def capture(cls) -> 'ResourceProfile':
        """Capture current system resource profile."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return cls(
            cpu_count=psutil.cpu_count(),
            cpu_percent=cpu_percent,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            disk_free=disk.free,
            load_average=os.getloadavg()
        )

    def can_allocate_workers(self, requested_workers: int) -> bool:
        """Check if system can handle requested number of workers."""
        # Reserve 2 CPUs for system overhead
        available_cpus = max(1, self.cpu_count - 2)

        # Check CPU load (keep below 80%)
        if self.cpu_percent > 80:
            available_cpus = max(1, available_cpus // 2)

        # Check memory (reserve 1GB for system)
        reserved_memory = 1 * 1024 * 1024 * 1024  # 1GB
        available_memory = self.memory_available - reserved_memory
        memory_per_worker = 100 * 1024 * 1024  # 100MB estimate per worker

        max_workers_by_memory = max(1, available_memory // memory_per_worker)

        return requested_workers <= min(available_cpus, max_workers_by_memory)


@dataclass
class GenerationTask:
    """Individual test generation task."""
    task_id: str
    module_name: str
    priority: str
    complexity: int
    estimated_time: float
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{int(time.time() * 1000)}_{self.module_name}"


@dataclass
class GenerationResult:
    """Result of a generation task."""
    task_id: str
    success: bool
    test_vectors: List[Any] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationCampaign:
    """Complete generation campaign with multiple tasks."""
    campaign_id: str
    tasks: List[GenerationTask]
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    results: List[GenerationResult] = field(default_factory=list)
    resource_profile: Optional[ResourceProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get campaign duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate of completed tasks."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return successful / len(self.results)

    @property
    def throughput(self) -> float:
        """Get tasks completed per second."""
        if self.duration <= 0:
            return 0.0
        return len(self.results) / self.duration


class ResourceMonitor:
    """Monitor system resources during generation."""

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.profiles: List[Tuple[float, ResourceProfile]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> List[Tuple[float, ResourceProfile]]:
        """Stop monitoring and return collected profiles."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.profiles.copy()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                profile = ResourceProfile.capture()
                self.profiles.append((time.time(), profile))
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break


class LoadBalancer:
    """Intelligent load balancing for generation tasks."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, GenerationTask] = {}
        self.completed_tasks: Dict[str, GenerationResult] = {}
        self.worker_assignments: Dict[int, str] = {}  # worker_id -> task_id

    def add_task(self, task: GenerationTask):
        """Add task to queue with priority-based ordering."""
        # Priority ordering: high > medium > low
        priority_map = {'high': 0, 'medium': 1, 'low': 2}
        priority_value = priority_map.get(task.priority, 1)

        # Include complexity and dependencies for smarter ordering
        queue_item = (priority_value, task.complexity, len(task.dependencies), task.task_id, task)
        self.task_queue.put(queue_item)

    def get_next_task(self, worker_id: int) -> Optional[GenerationTask]:
        """Get next task for worker, considering load balancing."""
        try:
            while not self.task_queue.empty():
                _, _, _, task_id, task = self.task_queue.get_nowait()

                # Check if task dependencies are satisfied
                if self._dependencies_satisfied(task):
                    self.active_tasks[task_id] = task
                    self.worker_assignments[worker_id] = task_id
                    return task
                else:
                    # Put back in queue if dependencies not ready
                    self.task_queue.put((_, _, _, task_id, task))

            return None
        except queue.Empty:
            return None

    def complete_task(self, worker_id: int, result: GenerationResult):
        """Mark task as completed and update load balancing state."""
        task_id = result.task_id
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        if worker_id in self.worker_assignments:
            del self.worker_assignments[worker_id]

        self.completed_tasks[task_id] = result

    def _dependencies_satisfied(self, task: GenerationTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep in task.dependencies:
            if dep not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep].success:
                return False
        return True

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queued': self.task_queue.qsize(),
            'active': len(self.active_tasks),
            'completed': len(self.completed_tasks),
            'workers_assigned': len(self.worker_assignments)
        }


class ParallelGenerationEngine:
    """
    High-performance parallel test generation engine.

    Features:
    - Intelligent resource management
    - Load balancing across workers
    - Fault tolerance and recovery
    - Real-time monitoring and reporting
    """

    def __init__(self, config):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.load_balancer: Optional[LoadBalancer] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.campaigns: Dict[str, GenerationCampaign] = {}

        # Performance tuning
        self.max_workers = self._calculate_optimal_workers()
        self.batch_size = getattr(config, 'batch_size', 50)
        self.timeout_per_task = 300  # 5 minutes per task

        logger.info(f"ParallelGenerationEngine initialized with {self.max_workers} max workers")

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        try:
            profile = ResourceProfile.capture()

            # Base calculation: CPU cores - 2 (reserve for system)
            optimal = max(1, profile.cpu_count - 2)

            # Adjust for memory (assume 200MB per worker)
            memory_per_worker = 200 * 1024 * 1024  # 200MB
            max_by_memory = max(1, profile.memory_available // memory_per_worker)

            # Adjust for CPU load
            if profile.cpu_percent > 70:
                optimal = max(1, optimal // 2)
            elif profile.cpu_percent > 90:
                optimal = 1  # Minimum when system is heavily loaded

            return min(optimal, max_by_memory, 8)  # Cap at 8 workers

        except Exception as e:
            logger.warning(f"Could not calculate optimal workers: {e}")
            return 2  # Conservative default

    def create_campaign(self, module_names: List[str], priorities: Dict[str, str] = None) -> str:
        """Create a new generation campaign."""
        campaign_id = f"campaign_{int(time.time())}"

        if priorities is None:
            priorities = {}

        # Create tasks for each module
        tasks = []
        for module_name in module_names:
            priority = priorities.get(module_name, 'medium')

            # Estimate complexity and time (simplified)
            complexity = self._estimate_module_complexity(module_name)
            estimated_time = complexity * 2.0  # Rough estimate

            task = GenerationTask(
                task_id="",
                module_name=module_name,
                priority=priority,
                complexity=complexity,
                estimated_time=estimated_time,
                metadata={'campaign_id': campaign_id}
            )
            tasks.append(task)

        campaign = GenerationCampaign(
            campaign_id=campaign_id,
            tasks=tasks,
            resource_profile=ResourceProfile.capture()
        )

        self.campaigns[campaign_id] = campaign
        logger.info(f"Created campaign {campaign_id} with {len(tasks)} tasks")
        return campaign_id

    def _estimate_module_complexity(self, module_name: str) -> int:
        """Estimate module complexity for load balancing."""
        # Simplified complexity estimation
        # In a real implementation, this would analyze the module
        base_complexity = 5

        # Adjust based on module name patterns
        if 'test' in module_name.lower():
            base_complexity += 2  # Test modules might be more complex
        if 'integration' in module_name.lower():
            base_complexity += 3  # Integration modules
        if len(module_name) > 20:
            base_complexity += 1  # Longer names might indicate complexity

        return min(base_complexity, 10)  # Cap complexity

    def execute_campaign(self, campaign_id: str, generation_func: Callable) -> Dict[str, Any]:
        """
        Execute a generation campaign with parallel processing.

        Args:
            campaign_id: Campaign to execute
            generation_func: Function that takes (module_name, context) and returns GenerationResult
        """
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign = self.campaigns[campaign_id]
        campaign.start_time = time.time()
        campaign.status = "running"

        logger.info(f"Starting campaign {campaign_id} with {self.max_workers} workers")

        # Initialize parallel execution
        self.load_balancer = LoadBalancer(self.max_workers)
        self.resource_monitor.start_monitoring()

        # Add all tasks to load balancer
        for task in campaign.tasks:
            self.load_balancer.add_task(task)

        # Execute with ThreadPoolExecutor for I/O bound generation
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch of tasks
            future_to_worker = {}
            for worker_id in range(self.max_workers):
                task = self.load_balancer.get_next_task(worker_id)
                if task:
                    future = executor.submit(self._execute_task, task, generation_func, worker_id)
                    future_to_worker[future] = worker_id

            # Process completed tasks and submit new ones
            while future_to_worker:
                for future in as_completed(future_to_worker, timeout=1.0):
                    worker_id = future_to_worker[future]

                    try:
                        result = future.result(timeout=self.timeout_per_task)
                        results.append(result)
                        self.load_balancer.complete_task(worker_id, result)

                        # Submit next task to this worker
                        next_task = self.load_balancer.get_next_task(worker_id)
                        if next_task:
                            next_future = executor.submit(self._execute_task, next_task, generation_func, worker_id)
                            future_to_worker[next_future] = worker_id

                    except Exception as e:
                        logger.error(f"Task execution failed for worker {worker_id}: {e}")
                        # Could implement retry logic here

                    del future_to_worker[future]
                    break

        # Campaign completion
        campaign.end_time = time.time()
        campaign.status = "completed"
        campaign.results = results

        # Stop monitoring
        resource_profiles = self.resource_monitor.stop_monitoring()

        # Generate campaign summary
        summary = self._generate_campaign_summary(campaign, resource_profiles)

        logger.info(f"Campaign {campaign_id} completed: {campaign.success_rate:.1%} success rate, "
                   f"{campaign.throughput:.2f} tasks/sec")

        return summary

    def _execute_task(self, task: GenerationTask, generation_func: Callable, worker_id: int) -> GenerationResult:
        """Execute a single generation task with monitoring."""
        start_time = time.time()

        try:
            # Capture resource usage before
            resources_before = self._capture_process_resources()

            # Execute generation function
            logger.debug(f"Worker {worker_id} executing task {task.task_id} for {task.module_name}")
            result_data = generation_func(task.module_name, task.metadata)

            # Capture resource usage after
            resources_after = self._capture_process_resources()

            execution_time = time.time() - start_time

            # Create result
            result = GenerationResult(
                task_id=task.task_id,
                success=True,
                test_vectors=result_data.get('vectors', []),
                execution_time=execution_time,
                resource_usage={
                    'cpu_time': resources_after['cpu_time'] - resources_before['cpu_time'],
                    'memory_peak': resources_after.get('memory_peak', 0),
                    'before': resources_before,
                    'after': resources_after
                },
                metadata=result_data.get('metadata', {})
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {e}")

            result = GenerationResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                resource_usage=self._capture_process_resources()
            )

        return result

    def _capture_process_resources(self) -> Dict[str, Any]:
        """Capture current process resource usage."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {
                'cpu_time': usage.ru_utime + usage.ru_stime,
                'memory_rss': usage.ru_maxrss * 1024,  # Convert to bytes
                'page_faults': usage.ru_majflt,
                'context_switches': usage.ru_nvcsw + usage.ru_nivcsw
            }
        except Exception:
            return {'cpu_time': 0, 'memory_rss': 0, 'page_faults': 0, 'context_switches': 0}

    def _generate_campaign_summary(self, campaign: GenerationCampaign,
                                 resource_profiles: List[Tuple[float, ResourceProfile]]) -> Dict[str, Any]:
        """Generate comprehensive campaign summary."""
        summary = {
            'campaign_id': campaign.campaign_id,
            'duration': campaign.duration,
            'tasks_total': len(campaign.tasks),
            'tasks_completed': len(campaign.results),
            'success_rate': campaign.success_rate,
            'throughput': campaign.throughput,
            'total_test_vectors': sum(len(r.test_vectors) for r in campaign.results if r.success),
            'average_execution_time': sum(r.execution_time for r in campaign.results) / max(1, len(campaign.results)),
            'resource_summary': self._summarize_resources(resource_profiles),
            'task_breakdown': self._analyze_task_performance(campaign.results),
            'performance_metrics': self._calculate_performance_metrics(campaign, resource_profiles)
        }

        return summary

    def _summarize_resources(self, profiles: List[Tuple[float, ResourceProfile]]) -> Dict[str, Any]:
        """Summarize resource usage throughout campaign."""
        if not profiles:
            return {}

        cpu_percentages = [p.cpu_percent for _, p in profiles]
        memory_percentages = [p.memory_percent for _, p in profiles]

        return {
            'cpu_avg': sum(cpu_percentages) / len(cpu_percentages),
            'cpu_peak': max(cpu_percentages),
            'memory_avg': sum(memory_percentages) / len(memory_percentages),
            'memory_peak': max(memory_percentages),
            'monitoring_points': len(profiles)
        }

    def _analyze_task_performance(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """Analyze performance by task characteristics."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            'successful': len(successful),
            'failed': len(failed),
            'avg_execution_time': sum(r.execution_time for r in results) / max(1, len(results)),
            'fastest_task': min((r.execution_time for r in successful), default=0),
            'slowest_task': max((r.execution_time for r in successful), default=0),
            'error_categories': self._categorize_errors(failed)
        }

    def _categorize_errors(self, failed_results: List[GenerationResult]) -> Dict[str, int]:
        """Categorize types of errors."""
        categories = {}
        for result in failed_results:
            # Simple error categorization
            error = result.error_message.lower()
            if 'timeout' in error:
                categories['timeout'] = categories.get('timeout', 0) + 1
            elif 'memory' in error:
                categories['memory'] = categories.get('memory', 0) + 1
            elif 'syntax' in error:
                categories['syntax'] = categories.get('syntax', 0) + 1
            else:
                categories['other'] = categories.get('other', 0) + 1
        return categories

    def _calculate_performance_metrics(self, campaign: GenerationCampaign,
                                     resource_profiles: List[Tuple[float, ResourceProfile]]) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        return {
            'resource_efficiency': self._calculate_resource_efficiency(campaign, resource_profiles),
            'scalability_score': self._calculate_scalability_score(campaign),
            'reliability_score': campaign.success_rate,
            'optimization_opportunities': self._identify_optimization_opportunities(campaign)
        }

    def _calculate_resource_efficiency(self, campaign: GenerationCampaign,
                                     resource_profiles: List[Tuple[float, ResourceProfile]]) -> float:
        """Calculate how efficiently resources were used."""
        if not campaign.results or not resource_profiles:
            return 0.0

        # Simple efficiency metric: throughput per average CPU usage
        avg_cpu = sum(p.cpu_percent for _, p in resource_profiles) / len(resource_profiles)
        if avg_cpu == 0:
            return campaign.throughput

        return campaign.throughput / (avg_cpu / 100.0)

    def _calculate_scalability_score(self, campaign: GenerationCampaign) -> float:
        """Calculate how well the system scaled."""
        # Simple scalability metric based on worker utilization
        if not self.load_balancer:
            return 0.5

        queue_status = self.load_balancer.get_queue_status()
        total_tasks = queue_status['queued'] + queue_status['active'] + queue_status['completed']

        if total_tasks == 0:
            return 0.5

        # Higher score if workers were well utilized
        utilization = queue_status['completed'] / total_tasks
        return min(1.0, utilization * 1.2)  # Slight bonus for completion

    def _identify_optimization_opportunities(self, campaign: GenerationCampaign) -> List[str]:
        """Identify potential optimization opportunities."""
        opportunities = []

        # Check for long-running tasks
        avg_time = sum(r.execution_time for r in campaign.results) / max(1, len(campaign.results))
        long_tasks = [r for r in campaign.results if r.execution_time > avg_time * 2]
        if long_tasks:
            opportunities.append(f"Optimize {len(long_tasks)} long-running tasks")

        # Check for failed tasks
        failed_count = sum(1 for r in campaign.results if not r.success)
        if failed_count > 0:
            opportunities.append(f"Address {failed_count} failed tasks")

        # Check for resource bottlenecks
        if campaign.resource_profile and campaign.resource_profile.cpu_percent > 80:
            opportunities.append("Consider reducing concurrent workers for CPU-bound workloads")

        return opportunities

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and capabilities."""
        profile = ResourceProfile.capture()

        return {
            'resource_profile': {
                'cpu_count': profile.cpu_count,
                'cpu_percent': profile.cpu_percent,
                'memory_percent': profile.memory_percent,
                'available_workers': min(self.max_workers, profile.cpu_count - 2)
            },
            'active_campaigns': len([c for c in self.campaigns.values() if c.status == 'running']),
            'completed_campaigns': len([c for c in self.campaigns.values() if c.status == 'completed']),
            'capabilities': {
                'parallel_processing': True,
                'resource_monitoring': True,
                'load_balancing': True,
                'fault_tolerance': True
            }
        }


# Convenience functions for easy integration
def create_parallel_campaign(module_names: List[str], config) -> str:
    """Create a parallel generation campaign."""
    engine = ParallelGenerationEngine(config)
    return engine.create_campaign(module_names)


def execute_parallel_generation(campaign_id: str, generation_func: Callable, config) -> Dict[str, Any]:
    """Execute parallel generation for a campaign."""
    engine = ParallelGenerationEngine(config)
    return engine.execute_campaign(campaign_id, generation_func)
