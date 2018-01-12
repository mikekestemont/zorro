import os
import glob
from collections import deque

""
class TripleStore(object):
    """Not used right now"""

    def __init__(self, input_, shingling='sentences',
                 allow_overlap=False, max_triples=None,
                 left_size=0, focus_size=0, right_size=0, shingle_stride=0):
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
        if shingling == 'characters':
            if not focus_size:
                raise ValueError(f'With character shingling option, focus_size should be >=1')
            if not left_size and not right_size:
                raise ValueError(f'At least, left_size OR right_size should be specified.')
        if not shingle_stride:
            shingle_stride = focus_size
        if shingle_stride > left_size + focus_size + right_size:
            raise ValueError(f'shingle_stride too large: you are missing out on data.')

        self.shingling = shingling
        self.focus_size = focus_size
        self.left_size = left_size
        self.right_size = right_size
        self.shingle_stride = shingle_stride
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
            chunk_size = self.left_size + self.focus_size + self.right_size
            for filename in self.filenames:
                chunk = ''
                for line in open(filename, 'r'):
                    line = line.strip()
                    if line:
                        chunk += ' ' + line
                    while len(chunk) >= chunk_size:
                        yield tuple([chunk[:self.left_size],
                                     chunk[self.left_size : self.left_size + self.focus_size],
                                     chunk[self.left_size + self.focus_size : -(len(chunk) - chunk_size)]])
                        self.processed += 1
                        if self.max_triples and self.processed >= self.max_triples:
                            return
                        if self.allow_overlap:
                            chunk = chunk[self.shingle_stride:]
                        else:
                            chunk = chunk[self.focus_size:]

    def __len__(self):
        # number of triples yielded so far
        return self.processed

class CoupleStore(object):

    def __init__(self, input_, shingling='sentences',
                 allow_overlap=False, max_couples=None,
                 focus_size=0, right_size=0, shingle_stride=0):
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
            raise ValueError(f'Unsupported `shingling` option: {shingling}')
        if shingling == 'characters':
            if not focus_size:
                raise ValueError(f'With "character" shingling, `focus_size` should be >=1')
            if not right_size:
                print('Setting `right_size` to that of `focus_size`')
                right_size = focus_size
        if not shingle_stride:
            print('Setting `shingle_stride` to `focus_size`')
            shingle_stride = focus_size
        if shingle_stride > right_size or shingle_stride > focus_size:
            raise ValueError(f'`shingle_stride` too large: you are missing out on data.')

        self.shingling = shingling
        self.focus_size = focus_size
        self.right_size = right_size
        self.shingle_stride = shingle_stride
        self.allow_overlap = allow_overlap
        self.max_couples = max_couples
        self.processed = 0

    def __iter__(self):
        if self.shingling == 'sentences':
            if self.allow_overlap:
                for filename in self.filenames:
                    couple = deque(maxlen=2)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            couple.append(line)
                            if len(couple) == 2:
                                self.processed += 1
                                yield tuple(couple)
                        if self.max_couples and self.processed >= self.max_couples:
                            return
            else:
                for filename in self.filenames:
                    couple = deque(maxlen=2)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            couple.append(line)
                            if len(couple) == 2:
                                self.processed += 1
                                yield tuple(couple)
                                couple = deque()
                        if self.max_couples and self.processed >= self.max_couples:
                            return

        elif self.shingling == 'characters':
            chunk_size = self.focus_size + self.right_size
            for filename in self.filenames:
                chunk = ''
                for line in open(filename, 'r'):
                    line = line.strip()
                    if line:
                        chunk += ' ' + line
                    while len(chunk) >= chunk_size:
                        yield tuple([chunk[:self.focus_size],
                                     chunk[self.focus_size : -(len(chunk) - chunk_size)]])
                        self.processed += 1
                        if self.max_couples and self.processed >= self.max_couples:
                            return
                        if self.allow_overlap:
                            chunk = chunk[self.shingle_stride:]
                        else:
                            chunk = chunk[self.focus_size + self.right_size:]

    def __len__(self):
        # number of triples yielded so far
        return self.processed

"""
def main():
    triple_store = TripleStore(input_='../data', shingling='characters',
                 size=100, allow_overlap=True, max_triples=10,
                 focus_size=21, left_size=21, right_size=21)
    for triple in triple_store:
        print('\t'.join([t for t in triple]))

if __name__ == '__main__':
    main()
"""
