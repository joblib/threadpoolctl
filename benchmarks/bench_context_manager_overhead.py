import time
from argparse import ArgumentParser
from pprint import pprint
from statistics import mean, stdev
from threadpoolctl import get_threadpool_limits, threadpool_limits

parser = ArgumentParser(description='Measure threadpool_limits call overhead.')
parser.add_argument('--import', dest="packages", nargs='+',
                    help='Python packages to import to load threadpool enabled'
                         ' libraries.')
parser.add_argument("--n-calls", type=int, default=100,
                    help="Number of iterations")

args = parser.parse_args()
for package_name in args.packages:
    __import__(package_name)

pprint(get_threadpool_limits())

timings = []
for _ in range(args.n_calls):
    t = time.time()
    with threadpool_limits(limits=1):
        pass
    timings.append(time.time() - t)

print(f"Overhead per call: {mean(timings) * 1000:.3f} "
      f"+/-{stdev(timings) * 1000:.3f} ms")
