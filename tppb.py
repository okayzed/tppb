#!/usr/bin/env python

# thibm partition balancer
# overview:
# create multiple initial partitioning methods
# pick the best partition candidate and optimize

import time
import math
import sys
import random
import copy

# {{{ TWEAKABLES
MAX_ITERATIONS=10000 # how many iterations to run (at max)
MAX_SECONDS=3 # how long before we stop trying to optimize a partitioning scheme
ITER_TIME=0.5 # how long to run before checking if local optima has been hit
STEP_SIZE=1 # a higher number means it runs faster but is less optimal

LOOP_BREAKER=True # break loops if we saw a partition that is continually being moved back and forth
MAX_LOOPS=50 # how many times before we consider a partition re-visiting its last index a loop

# probably don't change these
BEST_CANDIDATE_ONLY=True # only optimize the best candidate partition chosen
FAST_VALUATIONS=True # use a faster variance calculation that is partially memoized
DEBUG_SUMS=False # for debugging fast variance calculations
# }}}

# {{{ THIBM
class ThibM(list):
  # ThibM is a subclass of list (or array) and provides a function partition(K) to
  # split the ThibM into K partitions

  # {{{ INITIAL SPLITTING METHODS

  # adjusting left split walks from the left of the array towards the right,
  # making new partition boundaries when the current running sum goes over the
  # ideal bucket size. whenever a new partition is created, the ideal bucket
  # size is re-evaluated based on the remaining sum of the array
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

        # change our ideal bucket size
        avg_sum = prev_total / float(k_left)



      cur_sum += val


    return partitions

  # left split walks from the left of the array towards the right, making new
  # partition boundaries when the current running sum goes over the ideal
  # partition size. the ideal partition size does not change from its initial
  # value as sum(arr) / k
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

  # like adjusting left split, but from the right
  def adjusting_right_split(self, arr, k):
    rev_arr = list(reversed(arr))
    ret = self.adjusting_left_split(rev_arr, k)

    proper_indices = []
    for r in ret:
      proper_indices.append(len(arr) - r)


    return proper_indices

  # like constant left split, but from the right
  def right_split(self, arr, k):
    rev_arr = list(reversed(arr))
    ret = self.left_split(rev_arr, k)

    proper_indices = []
    for r in ret:
      proper_indices.append(len(arr) - r)

    return proper_indices

  # trying to do an adjusting split that starts from both ends.
  # NOTE: doesn't perform well at all, but is some indicator
  # of what a random split (or worse) might look like
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
  # }}} SPLITTING METHODS

  # {{{ HELPERS
  def print_stats(self, indeces, k, starting_std):
    final_std = self.evaluate_partitions(indeces, k)
    print "NEW INDECES:", indeces
    print "SUMS", map(sum, self.get_slices(indeces))
    print "TOTAL ITERATIONS:", self.iterations, "TOTAL MOVES:", self.moves
    print "FINAL STD:", final_std, "ORIGINAL:", starting_std
    print "IMPROVEMENT IS %.02f%% OF ORIGINAL" % (final_std / float(starting_std) * 100)

  # takes a partitioning scheme, supplied as an array of indeces
  # to split the array on and returns the K sub arrays specified
  # by those partition indeces.
  # [5] would create an array that is split with elements 0..5 and elements 6..N
  # [3, 5] would create 0..3, 4..5, 6..N
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


  # our prefix table lets us calculate the sum from arr[i] to arr[j] in
  # constant time, instead of O(j - i) time. this is useful for
  # calculating the sum inside each partition quickly
  def build_prefix_table(self):
    if not hasattr(self, 'prefix_sums'):
      self.prefix_sums = [0] * len(self)
      self.prefix_sums[0] = self[0]
      for i in xrange(1, len(self)):
        self.prefix_sums[i] = self.prefix_sums[i-1] + self[i]

  # returns a partition sum by consulting our prefix sum table
  # runs in O(1)
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

# }}} HELPERS

  # {{{ EVALUATION METHODS OF PARTITION VARIANCE

  # fast_evaluate calculates variance by maintaining the total sum of squares
  # and each individual partition's value sum. when a partition's boundary
  # changes, we call fast_evaluate and supply the partition index that changed.
  # the fast_evaluate function then re-calculates the partition's sum and its
  # neighbors, updating the total sum of squares with any changed values.
  # runtime of variance calculation is normally: O(N) where N=array size
  # this reduces runtime to O(1) for amortized variance calculations
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

  # calculate the variance of each partition's sum compared
  # to the avg partition sum (aka len(arr) / N)
  def evaluate_partitions(self, indeces, k, use_cache=False):
    arr = self
    self.build_prefix_table()
    sums = []

    # avg sum is just total sum divided by K, yay prefix sum tables
    avg_sum = self.prefix_sums[-1] / k
    delta = 0
    for p in xrange(len(indeces)):
      thisum = self.get_partition_sum(indeces, p)
      delta += (avg_sum - thisum)**2
    delta += (avg_sum - self.get_partition_sum(indeces, k-1))**2

    return math.sqrt(delta / k)

  # }}}

  # {{{ THE OPTIMIZATION FUNCTIONS
  # tries out multiple splitting methods and returns them in order of best fit
  def get_initial_candidate_partitions(self, k):
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
    return start_indeces

  # this is the main entry point for the partitioning algorithm
  # this function:
  # 1) gets likely partitioning schemes for data
  # 2) picks the best one and tries to optimize it until it can no longer optimize
  def partition(self, k):
    arr = self

    # returns multiple partition methods, best one first
    candidates = self.get_initial_candidate_partitions(k)
    self.results = []
    for data in candidates:
      score = data[0]
      indeces = data[1]
      method = data[2]
      print "USING METHOD", data[2]
      original_indeces = copy.copy(indeces)

      # the partition method is "failed" if it hasn't produced
      # the correct number of partition boundaries
      if len(indeces) != k - 1:
        print "LEN INDECES", len(indeces)
        print "PARTITIONING METHOD FAILED", method
        return

      # our best seen partition candidates so far
      best_val = sys.maxint
      best_indeces = copy.copy(indeces)


      # starting scores of the partition candidate as calculate by
      # our slow and fast evaluation schemes
      starting_std = self.evaluate_partitions(indeces, k)
      starting_faststd = self.fast_evaluate(indeces, k, [])
      print "STARTING STD", starting_std, "FAST", starting_faststd
      # storing state from last runs
      prev_val = starting_std # last iteration's score
      last_opt = starting_std # last local minima check's score (run every ITER_TIME seconds)


      last_move = {} # keeps track of every boundary move, used for finding loops
      loop_count = 0 # if our loop count goes above MAX_LOOPS, break out

      # maybe minima is an indicator to the outer loop to check to see if we've hit a minima.
      # conditions that can cause maybe minima being set to True:
      # 1) the move we just made has increased the score of the partition (instead of decreasing)
      # 2) we've been in a loop and re-visited the same index MAX_LOOP times
      # 3)
      maybe_minima = False

      start = time.time()
      prev = start

      self.iterations = 0
      self.moves = 0 # total number of partition movements we've made over all iterations

      # {{{ OUTER LOOP THAT KEEPS ITERATING UNTIL NO MORE OPTIMIZATIONS CAN BE MADE
      for i in xrange(MAX_ITERATIONS):
        self.iterations = i
        indeces.sort()

        # calculate our next likely moves by jittering around all the partition
        # boundaries
        movements = self.find_likely_permutation(indeces, k)

        moved = {} # what we've moved this iteration
        changes = 0 # how many changes we've made this iteration


        # make up to log(N) moves then re-calculate next moves
        num_moves = int(math.log(len(self)))

        # {{{ INNER LOOP THAT JIGGLES THE PARTITION BOUNDARIES
        for val,index,change in movements[:num_moves]:
          # only move each partition once per iteration
          if not index in moved:
            moved[index] = True
          else:
            continue

          now = time.time()
          self.moves += 1

          # if we've gone past ITER_TIME limit, check to see if we our score
          # has gone up or down since we last checked. if our score has not
          # gone down we break out and look for the best possible move to make.
          # if there are no moves, we call it a local minima
          if now - prev > ITER_TIME:
            print "ON ITERATION", i, "MOVES", self.moves, \
              "ESTIMATED IMPROVEMENT IS %.02f%% OF ORIGINAL" % (val / float(starting_std) * 100)
            prev = now
            print "PREV OPTIMUM", last_opt, "NEW OPT", val, "DELTA", (last_opt - val)

            if last_opt <= val:
              print "LIKELY LOCAL, STOPPING FOR NOW"
              maybe_minima = True
              break
            last_opt = val

          # if the move we just made has pushed us above our previous scores
          # that is an indication that we should break out of this moveset
          # and recalculate our list of next potential moves
          if prev_val <= val or last_opt <= val:
            print "REACHED LOCAL LOCAL MINIMA AFTER", len(moved), \
              "MOVES IN ITERATION %s, RECALCULATING MOVES" % (i)

            maybe_minima = True
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
              maybe_minima = True
              break

          last_move[index] = change
          changes += 1
        # }}} END OF INNER LOOP THAT MOVES THE PARTITIONS AROUND

        # if we've run past our time limit, exit no matter what
        if now - start > MAX_SECONDS:
          print "%s SECONDS HAVE ELAPSED, STOPPING" % MAX_SECONDS
          break

        # evaluate if we are in a local minima or not.
        # exit if we are
        if maybe_minima:
          # this is a last ditch effort to recover our scheme and
          # see if we can make any new moves that improve things
          movements = self.find_likely_permutation(indeces, k)
          val, index, change = movements[0]
          if index in last_move and last_move[index] + change == 0 and loop_count >= MAX_LOOPS:
            print "FORCE BREAKING LOOP"
            break

          if prev_val > val and last_opt > val and changes > 1:
            maybe_minima = False
            loop_count = 0
          else:
            break

        prev_val = val
      # }}} END OUTER LOOP

      # we are done with optimizing this partition, now print out some helpful stats
      # and info
      if maybe_minima:
        print "CONVERGED ON LOCAL OPTIMUM AFTER", i, "ITERATIONS AND", self.moves, "MOVES"
      else:
        print 'COULDNT CONVERGE ON OPTIMUM AFTER %s ITERATIONS AND %s SECONDS' % (self.iterations, int(now - start))

      val = self.evaluate_partitions(best_indeces, k)
      self.results.append([best_indeces, method, starting_std])
      self.results.append([original_indeces, "start " + method, starting_std])
      if not BEST_CANDIDATE_ONLY:
        self.print_stats(best_indeces, k, starting_std)

      if now - start > MAX_SECONDS:
        break

      if BEST_CANDIDATE_ONLY:
        break


  # for all partition boundaries in indeces, move the boundary up and down the
  # array and see how the variance changes. we greedily pick moves that will
  # lower our variance.
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

  # }}} OPTIMIZATION FUNCTIONS

# }}} THIBM

# {{{ TEST ARRAYS
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

# }}}  TEST DATASETS

# {{{ MAIN
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
# }}} MAIN
