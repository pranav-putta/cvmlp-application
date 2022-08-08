import math

import numpy as np

d_quad = lambda _a, _b, _x: 2 * _a * _x + _b

n = 100
approx_arc_length = lambda _a, _b: sum([(1 / n) * math.sqrt(1 + d_quad(_a, _b, _x)) for _x in
                                        np.arange(0, 1 + 1 / n, 1 / n)])  # approximate arc length from [0, 1]

print(approx_arc_length(1, 0))