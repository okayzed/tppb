import time
import math
import sys
import random

DEBUG_SUMS=False
MAX_ITERATIONS=10000
MAX_SECONDS=5


class ThibM(list):
  def adjusting_left_split(self, arr, k):
    overflow = 0
    prev_total = sum(arr)
    avg_sum = prev_total / float(k)

    cur_sum = 0
    partitions = []
    k_left = k
    for i in xrange(len(arr)):
      val = arr[i]
      prev_total -= val

      # time for a new partition
      if cur_sum + val > avg_sum:
        k_left -= 1
        if not k_left:
          break

        partitions.append(i)
        cur_sum -= avg_sum
        avg_sum = prev_total / float(k_left)



      cur_sum += val


    return partitions

  def left_split(self, arr, k):
    overflow = 0
    total_sum = sum(arr)
    avg_sum = total_sum / float(k)

    cur_sum = 0
    partitions = []
    for i in xrange(len(arr)):
      val = arr[i]


      # time for a new partition
      if cur_sum + val > avg_sum:
        partitions.append(i)
        cur_sum -= avg_sum

      cur_sum += val



    return partitions

  def adjusting_right_split(self, arr, k):
    rev_arr = list(reversed(arr))
    ret = self.adjusting_left_split(rev_arr, k)

    proper_indices = []
    for r in ret:
      proper_indices.append(len(arr) - r)


    return proper_indices

  def right_split(self, arr, k):
    rev_arr = list(reversed(arr))
    ret = self.left_split(rev_arr, k)

    proper_indices = []
    for r in ret:
      proper_indices.append(len(arr) - r)

    return proper_indices

  def split_array(self, arr, k):
    left_idx = 0
    right_idx = len(arr) - 1

    left_sum = 0
    right_sum = 0

    sum_total = sum(arr)

    bucket_size = sum_total / len(arr)
    indeces = []
    remaining_partitions = k
    remaining_sum = sum_total

    while left_idx < right_idx and remaining_partitions > 1:
      if arr[left_idx] > arr[right_idx]:
        remaining_sum -= arr[left_idx]
        left_sum += arr[left_idx]
        left_idx += 1
        if left_sum >= bucket_size:
          remaining_partitions -= 1
          bucket_size = max(remaining_sum / remaining_partitions, 1)
          indeces.append(left_idx)
          left_sum -= bucket_size
      else:
        remaining_sum -= arr[right_idx]
        right_sum += arr[right_idx]
        right_idx -= 1
        if right_sum >= bucket_size:
          remaining_partitions -= 1
          bucket_size = max(remaining_sum / remaining_partitions, 1)
          indeces.append(right_idx)
          right_sum -= bucket_size

    indeces.sort()
    return indeces

  def build_prefix_table(self):
    if not hasattr(self, 'prefix_sums'):
      self.prefix_sums = [0] * len(self)
      self.prefix_sums[0] = self[0]
      for i in xrange(1, len(self)):
        self.prefix_sums[i] = self.prefix_sums[i-1] + self[i]

  def get_slices(self, indeces):
    arr = self
    subarrs = []
    prev_index = 0
    for i in indeces:
      index = int(i)
      subarrs.append(arr[prev_index:index])
      prev_index = index

    if prev_index < len(arr):
      subarrs.append(arr[prev_index:])

    return subarrs

  def get_partition_sum(self, indeces, i):
    prev_index = 0

    if i > len(indeces):
      return 0

    if i > 0:
      prev_index = indeces[i-1] - 1

    if i == len(indeces):
      index = indeces[i-1]-1
      return self.prefix_sums[len(self)-1] - self.prefix_sums[index]


    index = indeces[i]-1
    return self.prefix_sums[index] - self.prefix_sums[prev_index]

  def evaluate_partitions(self, indeces, k, use_cache=False):
    arr = self
    self.build_prefix_table()
    sums = []

    avg_sum = self.prefix_sums[-1] / k
    delta = 0
    for p in xrange(len(indeces)):
      thisum = self.get_partition_sum(indeces, p)
      delta += (avg_sum - thisum)**2
    delta += (avg_sum - self.get_partition_sum(indeces, k-1))**2

    return math.sqrt(delta / k)

  def partition(self, k):
    arr = self

    start_indeces = []
    methods = "adjusting left", "constant left split", "adjusting right", "constant right split", "adjusting center split"
    for i,func in enumerate([self.adjusting_left_split, self.left_split, self.adjusting_right_split, self.right_split, self.split_array]):
      indeces = func(arr, k)

      # VERY IMPORTANT TO SORT INDECES
      indeces.sort()


      val = self.evaluate_partitions(indeces, k)
      name = methods[i]

      if len(indeces) < k-1:
        print "COULDNT SPLIT INITIAL ARRAY WITH METHOD", methods[i], "INDECES ARE", len(indeces)
      else:
        indeces = indeces[:k-1]
        print "INITIAL SPLIT FROM %s HAS STD OF %s" % (name, val)

        start_indeces.append([val, indeces, name])

    start_indeces.sort()

    print "USING METHOD", start_indeces[0][2]
    indeces = start_indeces[0][1]

    self.indeces = indeces
    if len(indeces) != k - 1:
      print "LEN INDECES", len(indeces)
      print "PARTITIONING METHOD FAILED", methods[i]
      return

    starting = self.evaluate_partitions(indeces, k)

    prev_val = starting
    last_opt = starting
    threshold = 1
    success = False
    self.starting_std = self.evaluate_partitions(indeces, k)
    print "STARTING STD", self.starting_std
    start = time.time()
    prev = start
    moves = 0
    for i in xrange(MAX_ITERATIONS):
      self.iterations = i
      indeces.sort()
      movements = self.find_likely_permutation(indeces, k)
      moved = {}

      num_moves = int(math.log(len(self)))
      for val,index,change in movements[:num_moves]:
        now = time.time()
        moves += 1

        if not index in moved:
          moved[index] = True
        else:
          continue


        if now - prev > 1:
          print "ON ITERATION", i, "MOVES", moves, "IMPROVEMENT IS %.02f%% OF ORIGINAL" % (val / float(self.starting_std) * 100)
          prev = now
          print "PREV OPTIMUM", last_opt, "NEW OPT", val, "DELTA", (last_opt - val)

          if last_opt == val:
            print "LIKELY LOCAL, STOPPING FOR NOW"
            success = True
            break
          last_opt = val

        indeces[index] += change
        if prev_val <= val:
          print "REACHED LOCAL LOCAL MINIMA AFTER", len(moved), "MOVES IN ITERATION %s, RECALCULATING MOVES" % (i)

          success = True
          break


      if now - start > MAX_SECONDS:
        print "%s SECONDS HAVE ELAPSED, STOPPING" % MAX_SECONDS
        break

      if success:
        movements = self.find_likely_permutation(indeces, k)
        val, index, change = movements[0]
        indeces[index] += change
        val = self.evaluate_partitions(indeces, k)
        if prev_val > val:
          success = False
        else:
          break
      prev_val = val

    if success:
      print "CONVERGED ON LOCAL OPTIMUM AFTER", i, "ITERATIONS AND", moves, "MOVES"
    else:
      print 'COULDNT CONVERGE ON OPTIMUM AFTER %s ITERATIONS AND %s SECONDS' % (self.iterations, int(now - start))


  # for any set of indeces, figure out
  # what moving each partition will do to the global values
  # types of movement:
  #   for all pairs of indeces, evaluate each index pair with offsets applied
  def find_likely_permutation(self, indeces, k):
    movements = []

    steps = 3
    self.evaluate_partitions(indeces, k)
    for i, pos in enumerate(indeces):
      min_pos = 0
      if i > 0:
        min_pos = (indeces[i-1]+1 + indeces[i]) / 2

      max_pos = len(self) - 1
      if i < len(indeces) - 1:
        max_pos = (indeces[i+1]-1 + indeces[i]) / 2

      for j in xrange(min_pos, max_pos, steps):
        indeces[i] = j
        val = self.evaluate_partitions(indeces, k)
        movements.append([val, i, j-pos])

      indeces[i] = pos


    movements.sort()
    return movements

  def print_stats(self, k):
    final_std = self.evaluate_partitions(self.indeces, k)
    print "NEW INDECES:", self.indeces
    print "SUMS", map(sum, self.get_slices(self.indeces))
    print "TOTAL ITERATIONS:", self.iterations
    print "FINAL STD:", final_std
    print "IMPROVEMENT IS %.02f%% OF ORIGINAL" % (final_std / float(self.starting_std) * 100)


def uniform_arr(size):
  ret = []
  for i in xrange(size):
    ret.append(random.randint(0, 10000))

  return ret

def mixed_arr(size):
  ret = []
  big_nums = int(math.log(size))
  smalls = size - big_nums
  for i in xrange(size):
    ret.append(random.randint(0, 1000))

  for i in xrange(big_nums):
    ret.append(random.randint(100000, 1000000))

  return ret

def expo_numbers(size):
  ret = []
  for i in xrange(size):
    ret.append(random.expovariate(3))

  return ret
def uniform_big_numbers(size):
  ret = []
  for i in xrange(size):
    ret.append(random.randint(0, sys.maxint))

  return ret


def main():

  # reopen stdout unbuffered
  import os
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

  n = 1000
  k = 10

  if len(sys.argv) >= 3:
    n = int(sys.argv[1])
    k = int(sys.argv[2])

  methods = ["SEQUENTIAL", "UNIFORM", "MIXED", "EXPO", "UNIFORM BIG NUM"]
  for i, gen in enumerate([ range, uniform_arr, mixed_arr, expo_numbers, uniform_big_numbers ]):
    l = gen(n)
    method = methods[i]
    print "GENERATING %s %s SAMPLES" % (n, method)
    print "CREATING %s PARTITIONS" % k

    t = ThibM(l)

    try:
      t.partition(k)
    except KeyboardInterrupt:
      print "Keyboard Interrupted"

    t.print_stats(k)
    print "\n\n\n"

if __name__ == "__main__":
  main()
