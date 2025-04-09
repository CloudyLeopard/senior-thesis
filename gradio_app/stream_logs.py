import gradio as gr
import logging
from queue import Queue, Empty
import threading
import time

# -------------------------------
# Setup Logging to a Queue
# -------------------------------

# Create a thread-safe queue for log messages.
log_queue = Queue()

# Custom logging handler that puts log messages into the queue.
class QueueLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_queue.put(msg)
        except Exception:
            self.handleError(record)

# -------------------------------
# Optimized Log Streaming Generator
# -------------------------------


def stream_logs():
    accumulated_logs = [f"Begin research session {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    yield accumulated_logs[0]

    while True:
        # Block until at least one log message is available.
        new_log = log_queue.get(block=True)
        accumulated_logs.append(new_log)
        
        # Retrieve any additional logs without blocking.
        while True:
            try:
                accumulated_logs.append(log_queue.get_nowait())
            except Empty:
                break
        
        # Join the new logs (you may want to yield the entire history if required)
        full_log = "\n".join(accumulated_logs) + "\n"
        yield full_log

        # Reset the accumulated list if you do not need to keep the full history.
        # This prevents repeated yields of the same logs.
        # accumulated_logs = []

# -------------------------------
# Logger Setup
# -------------------------------

# default logger setup
logging.basicConfig(level=logging.WARNING)

# Define custom queue handler to the logger.
queue_handler = QueueLoggingHandler()
queue_handler.setFormatter(logging.Formatter("%(name)s - %(asctime)s - %(levelname)s - %(message)s"))

# Setup the librarian logger
logger_librarian = logging.getLogger("kruppe.algorithm.librarian")
logger_librarian.setLevel(logging.DEBUG)
# logger_librarian.propagate = False  # Ensure only messages from this logger are processed.
logger_librarian.addHandler(queue_handler) # Add the queue handler to the logger.

# Set up WebScraper logger
logger_librarian = logging.getLogger("kruppe.data_source.utils")
logger_librarian.setLevel(logging.INFO)
# logger_librarian.propagate = False 
logger_librarian.addHandler(queue_handler)

# # -------------------------------
# # Simulated Log Generation (Optional)
# # -------------------------------

# # This function is only for demonstration purposes.
# def generate_logs():
#     while True:
#         logger.info("A new log entry at " + time.strftime("%H:%M:%S"))
#         time.sleep(2)

# # If you already have logging happening naturally in your application,
# # you can remove this thread.
# log_thread = threading.Thread(target=generate_logs, daemon=True)
# log_thread.start()

# -------------------------------
# Gradio Interface
# -------------------------------

def create_log_interface():
    iface = gr.Interface(
        fn=stream_logs,
        inputs=None,
        outputs=[gr.Textbox(lines=30, label="Log Output")],
        live=True,  # Enables streaming output.
        flagging_mode='manual',
        flagging_dir='logs',
        clear_btn=None,
        show_progress='hidden',
        title="Log Stream"
    )
    return iface

