import asyncio
import threading
import time
from typing import Optional

# Background hunter state
_hunter_thread: Optional[threading.Thread] = None
_hunter_running: bool = False


def start_background_jobs(log):
    """Start background jobs including heartbeat and proactive hunter."""
    
    def heartbeat():
        while True:
            log.info("python service heartbeat")
            time.sleep(60)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    
    # Start proactive background hunter
    start_proactive_hunter(log)


def start_proactive_hunter(log):
    """Start the proactive background data hunter (every 30 minutes)."""
    global _hunter_thread, _hunter_running
    
    if _hunter_thread is not None and _hunter_thread.is_alive():
        log.info("Proactive hunter already running")
        return
    
    def hunter_loop():
        global _hunter_running
        _hunter_running = True
        
        # Import here to avoid circular imports
        from .crawler import get_background_hunter
        
        log.info("Starting proactive background hunter (15 min interval)")
        
        # Wait 5 minutes before first hunt to let system stabilize
        time.sleep(300)
        
        hunter = get_background_hunter()
        
        while _hunter_running:
            try:
                log.info("Running proactive hunt cycle...")
                
                # Run async hunt in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    for target in hunter.hunt_targets:
                        try:
                            from .crawler import autonomous_hunt
                            result = loop.run_until_complete(autonomous_hunt(target))
                            
                            log.info(f"Proactive hunt '{target[:50]}': {result.get('items_collected', 0)} items")
                            
                            hunter.hunt_results.append({
                                "target": target,
                                "items": result.get("items_collected", 0),
                                "sources": result.get("sources", []),
                            })
                            
                        except Exception as e:
                            log.error(f"Proactive hunt target error: {e}")
                finally:
                    loop.close()
                
                from datetime import datetime
                hunter.last_hunt = datetime.now()
                log.info(f"Proactive hunt cycle complete. Next in 15 minutes.")
                
            except Exception as e:
                log.error(f"Proactive hunter cycle error: {e}")
            
            # Wait 15 minutes
            time.sleep(15 * 60)
    
    _hunter_thread = threading.Thread(target=hunter_loop, daemon=True)
    _hunter_thread.start()
    log.info("Proactive background hunter thread started")


def stop_proactive_hunter(log):
    """Stop the proactive background hunter."""
    global _hunter_running
    _hunter_running = False
    log.info("Proactive hunter stop requested")
