import queue
import threading
from collections import defaultdict
from typing import Dict, Any, List

class SimulatedKafkaBroker:
    """
    An in-memory, thread-safe message broker mimicking Kafka publish-subscribe topic architecture.
    Allows multiple producers to publish to topics and multiple consumers to read messages asynchronously.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern so all threads connect to the exact same global broker."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(SimulatedKafkaBroker, cls).__new__(cls, *args, **kwargs)
                cls._instance._init_broker()
            return cls._instance

    def _init_broker(self):
        # Maps topic name -> list of subscriber queues
        self.subscribers: Dict[str, List[queue.Queue]] = defaultdict(list)
        self.topic_lock = threading.Lock()
        print("[Kafka Broker] In-memory message broker initialized successfully.")

    def publish(self, topic: str, message: Dict[str, Any]):
        """
        Publish a message to a topic. The message will be copied into 
        the queues of all active subscribers to this topic.
        """
        with self.topic_lock:
            subs = self.subscribers[topic]
            if not subs:
                # No active consumer listening, drop message or let it wait
                return
                
            for sub_queue in subs:
                try:
                    # Put message into subscriber queue (non-blocking)
                    sub_queue.put_nowait(message.copy())
                except queue.Full:
                    # If consumer queue is full, discard oldest message to maintain low latency (FIFO pruning)
                    try:
                        sub_queue.get_nowait()
                        sub_queue.put_nowait(message.copy())
                    except Exception:
                        pass

    def subscribe(self, topic: str) -> queue.Queue:
        """
        Subscribe to a topic. Returns a thread-safe Queue object from which 
        the consumer can continuously pull incoming events.
        """
        with self.topic_lock:
            # Create a dedicated queue for this subscription with a max capacity of 5000 records
            sub_queue = queue.Queue(maxsize=5000)
            self.subscribers[topic].append(sub_queue)
            print(f"[Kafka Broker] Consumer subscribed to topic: '{topic}' (active subscribers: {len(self.subscribers[topic])})")
            return sub_queue

    def unsubscribe(self, topic: str, sub_queue: queue.Queue):
        """Unsubscribe and release the consumer queue."""
        with self.topic_lock:
            if topic in self.subscribers:
                self.subscribers[topic].remove(sub_queue)
                print(f"[Kafka Broker] Consumer unsubscribed from topic: '{topic}'")
