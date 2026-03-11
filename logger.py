
import os
import sys
import time


class TextLogger(object):

    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
 

    def __init__(self, root, phase="train"):
        self.root = root
        self.phase = phase
        self.epoch = 0

        os.makedirs(self.root, exist_ok=True)

        # redirect std out
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger

    def _get_phase_or_epoch(self):
        if self.phase == "train":
            return str(self.epoch)
        else:
            return self.phase

    def close(self):
        self.logger.close()
