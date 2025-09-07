import math
import random


# Floating-point comparison functions
def is_equal(a, b, tol=1e-6):
    return math.isclose(a, b, abs_tol=tol)


def is_less(a, b, tol=1e-6):
    return a < b and not is_equal(a, b, tol)


def is_greater(a, b, tol=1e-6):
    return a > b and not is_equal(a, b, tol)


def is_less_equal(a, b, tol=1e-6):
    return is_less(a, b, tol) or is_equal(a, b, tol)


def is_greater_equal(a, b, tol=1e-6):
    return is_greater(a, b, tol) or is_equal(a, b, tol)


# Random Boolean Generator
class RandomBoolGenerator:
    def __init__(self):
        self.sequence = None
        self.gen = random.Random()
        self.dist = lambda: self.gen.getrandbits(32)

    def __call__(self):
        if self.sequence is None:
            self.sequence = self.dist()
        result = self.sequence & 1
        self.sequence >>= 1
        return bool(result)

def overlap_penalty(blocks, penalty=1e6):
    total_penalty = 0.0
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            if (b1.get_min_x() < b2.get_max_x() and b1.get_max_x() > b2.get_min_x() and
                b1.get_min_y() < b2.get_max_y() and b1.get_max_y() > b2.get_min_y()):
                total_penalty += penalty
    return total_penalty

# Cost Function
class PA3Cost:
    def __init__(self, expected_aspect_ratio):
        self.expected_aspect_ratio = expected_aspect_ratio

    def __call__(self, tree):
        w = tree.get_width()
        h = tree.get_height()
        area = tree.get_area()
        total_hpwl = tree.get_total_hpwl()
        aspect_ratio = max(w / h, h / w)
        ratio_diff_3 = (aspect_ratio - self.expected_aspect_ratio) / 3.0
        cost = (0.25 * area + 0.75 * total_hpwl) * (1 + ratio_diff_3 ** 2)
        cost += overlap_penalty(tree.blocks) * 10

        return cost


import time

class Timer:
    def __init__(self, timeout):
        self.start = time.time()
        self.timeout = timeout

    def is_timeout(self):
        return (time.time() - self.start) > self.timeout