import threading
import time


def start_background_jobs(log):
    def heartbeat():
        while True:
            log.info("python service heartbeat")
            time.sleep(60)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
