import os
import glob
from collections import deque


class TripleStore(object):

    def __init__(self, input_, shingling='sentences',
                 size=100, allow_overlap=False, max_triples=None):
        """
        size is the number of characters in a single when `shingling=characters.
        """
        if os.path.isdir(input_):
            if not input_.endswith('/'):
                input_ += '/'
            self.filenames = sorted(list(glob.glob(input_ + '*.txt')))
        else:
            self.filenames = [input_]

        if shingling not in ('sentences', 'characters'):
            raise ValueError(f'Unsupported shingling option {shingling}')

        self.shingling = shingling
        self.allow_overlap = allow_overlap
        self.max_triples = max_triples
        self.processed = 0

    def __iter__(self):
        if self.shingling == 'sentences':
            if self.allow_overlap:
                for filename in self.filenames:
                    triple = deque(maxlen=3)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            triple.append(line)
                            if len(triple) == 3:
                                self.processed += 1
                                yield tuple(triple)
                        if self.max_triples and self.processed >= self.max_triples:
                            return
            else:
                for filename in self.filenames:
                    triple = deque(maxlen=3)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            triple.append(line)
                            if len(triple) == 3:
                                self.processed += 1
                                yield tuple(triple)
                                triple = deque()
                        if self.max_triples and self.processed >= self.max_triples:
                            return

        elif self.shingling == 'characters':
            raise NotImplementedError

    def __len__(self):
        # number of triples yielded so far
        return self.processed
