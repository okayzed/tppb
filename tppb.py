import time
import math
import sys
import random
import copy

DEBUG_SUMS=False
MAX_ITERATIONS=10000
MAX_SECONDS=3
BEST_CANDIDATE_ONLY=True
LOOP_BREAKER=True
MAX_LOOPS=50
FAST_VALUATIONS=True
STEP_SIZE=1
ITER_TIME=0.5

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

  def center_split_array(self, arr, k):
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
          left_sum -= bucket_size
          indeces.append(left_idx)
          indeces.sort()
      else:
        remaining_sum -= arr[right_idx]
        right_sum += arr[right_idx]
        right_idx -= 1
        if right_sum >= bucket_size:
          remaining_sum -= right_sum
          remaining_partitions -= 1
          right_sum -= bucket_size
          indeces.append(right_idx+1)
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

    if prev_index < len(arr) - 1 and len(indeces) > 0:
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

  def fast_evaluate(self, indeces, k, adjusted=[]):
    if not hasattr(self, "fast_sums_cache") or False:
      self.fast_sum_total = 0
      self.fast_sums_cache = {}
      adjusted = range(len(indeces))

    avg_sum = self.prefix_sums[-1] / k
    new_values = {}
    for i in adjusted:
      if i not in new_values:
        new_values[i] = self.get_partition_sum(indeces, i)
      if i-1 >= 0 and i-1 not in new_values:
        new_values[i-1] = self.get_partition_sum(indeces, i-1)
      if i+1 < len(indeces) and i+1 not in new_values:
        new_values[i+1] = self.get_partition_sum(indeces, i+1)

    for index in new_values:
      new_val = (avg_sum - new_values[index])**2
      old_val = 0

      if index in self.fast_sums_cache:
        old_val = self.fast_sums_cache[index]

      self.fast_sums_cache[index] = new_val

      self.fast_sum_total -= old_val
      self.fast_sum_total += new_val


    return math.sqrt(self.fast_sum_total / k)




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
    for i,func in enumerate([self.adjusting_left_split, self.left_split, self.adjusting_right_split, self.right_split, self.center_split_array]):
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

    self.results = []
    for data in start_indeces:
        print "USING METHOD", data[2]
        indeces = data[1]
        original_indeces = copy.copy(indeces)
        best_val = sys.maxint
        best_indeces = copy.copy(indeces)

        if len(indeces) != k - 1:
          print "LEN INDECES", len(indeces)
          print "PARTITIONING METHOD FAILED", methods[i]
          return

        starting = self.evaluate_partitions(indeces, k)

        prev_val = starting
        last_opt = starting
        last_move = {}
        threshold = 1
        loop_count = 0
        success = False
        starting_std = self.evaluate_partitions(indeces, k)
        starting_faststd = self.fast_evaluate(indeces, k, [])
        print "STARTING STD", starting_std, "FAST", starting_faststd
        start = time.time()
        prev = start
        moves = 0


        for i in xrange(MAX_ITERATIONS):
          self.iterations = i
          indeces.sort()
          movements = self.find_likely_permutation(indeces, k)
          moved = {}
          changes = 0

          num_moves = int(math.log(len(self)))
          for val,index,change in movements[:num_moves]:
            if not index in moved:
              moved[index] = True
            else:
              continue


            now = time.time()
            moves += 1
            if now - prev > ITER_TIME:
              print "ON ITERATION", i, "MOVES", moves, \
                "ESTIMATED IMPROVEMENT IS %.02f%% OF ORIGINAL" % (val / float(starting_std) * 100)
              prev = now
              print "PREV OPTIMUM", last_opt, "NEW OPT", val, "DELTA", (last_opt - val)

              if last_opt <= val:
                print "LIKELY LOCAL, STOPPING FOR NOW"
                success = True
                break
              last_opt = val

            if prev_val <= val or last_opt <= val:
              print "REACHED LOCAL LOCAL MINIMA AFTER", len(moved), \
                "MOVES IN ITERATION %s, RECALCULATING MOVES" % (i)

              success = True
              break

            # dont change the indeces until we are sure it improves things
            indeces[index] += change
            if val < best_val:
              best_val = val
              best_indeces = copy.copy(indeces)

            if LOOP_BREAKER and index in last_move and last_move[index] + change == 0:
              # we need to recognize this likely loop and if nothing else has changed...
              loop_count += 1
              if loop_count >= MAX_LOOPS:
                print "LIKELY LOOP, BREAKING THE CYCLE"
                success = True
                break

            last_move[index] = change
            changes += 1



          if now - start > MAX_SECONDS:
            print "%s SECONDS HAVE ELAPSED, STOPPING" % MAX_SECONDS
            break

          if success:
            movements = self.find_likely_permutation(indeces, k)
            val, index, change = movements[0]
            if index in last_move and last_move[index] + change == 0 and loop_count >= MAX_LOOPS:
              print "FORCE BREAKING LOOP"
              break

            if prev_val > val and last_opt > val and changes > 1:
              success = False
              loop_count = 0
            else:
              break

            # dont make change unless we think it really makes things better
            indeces[index] += change
            val = self.evaluate_partitions(indeces, k)

          prev_val = val

        if now - start > MAX_SECONDS:
          break
        if success:
          print "CONVERGED ON LOCAL OPTIMUM AFTER", i, "ITERATIONS AND", moves, "MOVES"
        else:
          print 'COULDNT CONVERGE ON OPTIMUM AFTER %s ITERATIONS AND %s SECONDS' % (self.iterations, int(now - start))

        val = self.evaluate_partitions(best_indeces, k)
        self.results.append([best_indeces, name, starting_std])
        self.results.append([original_indeces, "start " + name, starting_std])
        if not BEST_CANDIDATE_ONLY:
          self.print_stats(best_indeces, k, starting_std)

        if BEST_CANDIDATE_ONLY:
          break


  # for any set of indeces, figure out
  # what moving each partition will do to the global values
  # types of movement:
  #   for all pairs of indeces, evaluate each index pair with offsets applied
  def find_likely_permutation(self, indeces, k):
    movements = []

    steps = STEP_SIZE
    self.evaluate_partitions(indeces, k)
    for i, pos in enumerate(indeces):
      min_pos = 0
      if i > 0:
        min_pos = (indeces[i-1]+1 + indeces[i]) / 2

      max_pos = len(self) - 1
      if i < len(indeces) - 1:
        max_pos = (indeces[i+1]-1 + indeces[i]) / 2

      for j in xrange(min_pos, max_pos, steps):
        if j == pos:
          continue

        indeces[i] = j
        if FAST_VALUATIONS:
          fastval = self.fast_evaluate(indeces, k, [i])
          movements.append([fastval, i, j-pos])
        else:
          val = self.evaluate_partitions(indeces, k)
          movements.append([val, i, j-pos])

      self.fast_evaluate(indeces, k, [i])
      indeces[i] = pos


    movements.sort()

    return movements

  def print_stats(self, indeces, k, starting_std):
    final_std = self.evaluate_partitions(indeces, k)
    print "NEW INDECES:", indeces
    print "SUMS", map(sum, self.get_slices(indeces))
    print "TOTAL ITERATIONS:", self.iterations
    print "FINAL STD:", final_std, "ORIGINAL:", starting_std
    print "IMPROVEMENT IS %.02f%% OF ORIGINAL" % (final_std / float(starting_std) * 100)


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
  twice_interrupted = False
  for i, gen in enumerate([ range, uniform_arr, mixed_arr, expo_numbers, uniform_big_numbers ]):
    l = gen(n)
    method = methods[i]
    print "GENERATING %s %s SAMPLES" % (n, method)
    print "CREATING %s PARTITIONS" % k

    start = time.time()
    t = ThibM(l)

    try:
      t.partition(k)
    except KeyboardInterrupt:
      print "Keyboard Interrupted, Interrupt again to exit fully"
      if twice_interrupted:
        break
      twice_interrupted = True

    t.results.sort(key=lambda x: t.evaluate_partitions(x[0], k))
    data = t.results[0]
    print "BEST RESULTS %s" % data[1]
    now = time.time()
    t.print_stats(data[0], k, data[2])
    print "TOTAL TIME: %s seconds" % (now - start)
    print "\n\n\n"

if __name__ == "__main__":
  main()
