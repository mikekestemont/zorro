from datetime import datetime
import json
import logging

from seqmod.misc.loggers import Logger, StdLogger


class JsonLogger(StdLogger):
    """
    Standard python logger.
    Parameters
    ----------
    - outputfile: str, file to print log to. If None, only a console
        logger will be used.
    - level: str, one of 'INFO', 'DEBUG', ... See logging.
    - msgfmt: str, message formattter
    - datefmt: str, date formatter
    """
    def __init__(self, outputfile=None, level='INFO', json_file=None,
                 msgfmt="[%(asctime)s] %(message)s", datefmt='%m-%d %H:%M:%S'):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.handlers = []
        self.logger.setLevel(getattr(logging, level))
        formatter = logging.Formatter(msgfmt, datefmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if outputfile is not None:
            fh = logging.FileHandler(outputfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if json_file:
            now = '{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) + '_'
            self.json_file = now + json_file
            self.history = []

    def checkpoint(self, payload):
        e, b, bs = payload['epoch'], payload['batch'], payload['total_batches']
        speed = payload["examples"] / payload["duration"]
        loss = StdLogger.loss_str(payload['loss'], 'train')
        self.logger.info("Epoch[%d]; batch [%d/%d]; %s; speed %d tokens/sec" %
                         (e, b, bs, loss, speed))
        if self.json_file:
            payload['dataset'] = 'train'
            self.history.append(payload)
            with open(self.json_file, 'w') as jf:
                jf.write(json.dumps(self.history, indent=4))

    def validation_end(self, payload):
        loss = StdLogger.loss_str(payload['loss'], 'valid')
        self.logger.info("Epoch[%d]; %s" % (payload['epoch'], loss))
        if self.json_file:
            payload['dataset'] = 'valid'
            self.history.append(payload)
            with open(self.json_file, 'w') as jf:
                jf.write(json.dumps(self.history, indent=4))
