"""Check tests are not skipped in every ci job"""

from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET


base_dir = sys.argv[1]

# dict {test: result} where result is False if the test was skipped in every
# job and True otherwise.
always_skipped = {}

for name in os.listdir(base_dir):
    # all test result files are in /base_dir/jobs.*/ dirs
    if name.startswith("stage1."):
        print("> processing test result from job", name.replace("stage1", ""))
        print("  > tests skipped:")
        result_file = os.path.join(base_dir, name, "test-data.xml")
        root = ET.parse(result_file).getroot()

        # All tests are identified by the xml tag testcase.
        for test in root.iter("testcase"):
            test_name = test.attrib["name"]
            skipped = any(child.tag == "skipped" for child in test)
            if skipped:
                print("    -", test_name)
            if test_name in always_skipped:
                always_skipped[test_name] &= skipped
            else:
                always_skipped[test_name] = skipped

print("\n------------------------------------------------------------------\n")
fail = False
for test, skipped in always_skipped.items():
    if skipped:
        fail = True
        print(test, "was skipped in every job")

if fail:
    sys.exit(1)
