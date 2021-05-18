#!/usr/bin/env python3

from __future__ import print_function
import os
import optparse
import random
import errno
import struct
import perfmon

if __name__ == '__main__':
  parser = optparse.OptionParser()
  parser.add_option("-e", "--events", help="Events to use",
                    action="store", dest="events")
  parser.set_defaults(events="PERF_COUNT_HW_CPU_CYCLES")
  (options, args) = parser.parse_args()

  if options.events:
    events = options.events.split(",")
  else:
    raise Exception("You need to specify events to monitor")

  events = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'UNHALTED_REFERENCE_CYCLES', \
        'LLC_MISSES', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_CPU_CYCLES', 'PERF_COUNT_HW_BRANCH_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I', 'UOPS_RETIRED']

  s = perfmon.PerThreadSession(int(os.getpid()), events)
  s.start()

  # code to be measured
  #
  # note that this is not identical to what examples/self.c does
  # thus counts will be different in the end
  for i in range(1, 1000000):
    random.random()

  # read the counts
  for i in range(0, len(events)):
    count = struct.unpack("L", s.read(i))[0]
    print("""%s\t%lu""" % (events[i], count))