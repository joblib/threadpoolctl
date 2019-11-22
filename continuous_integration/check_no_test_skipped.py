"""Check tests are not skipped in every ci job"""

from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET


base_dir = sys.argv[1]

# dict {test: result} where result is False if the test was skipped in every
# job and True otherwise.
aggregated_results = {}

for name in os.listdir(base_dir):
    # all test result files are in /base_dir/jobs.*/ dirs
    if name.startswith("stage1."):
        print("> processing test result from job", name.replace("stage1", ""))
        result_file = os.path.join(base_dir, name, "test-data.xml")
        root = ET.parse(result_file).getroot()

        # All tests are identified by the xml tag testcase.
        for test in root.iter("testcase"):
            test_name = test.attrib["name"]
            if test_name not in aggregated_results:
                aggregated_results[test_name] = False
            print("  -", test_name)

            for child in test:
                print("    -", child.tag)
                if child.tag == "skipped":
                    aggregated_results[test_name] |= False
            else:
                aggregated_results[test_name] |= True

print("\n------------------------------------------------------------------\n")
fail = False
for test, result in aggregated_results.items():
    if not result:
        fail = True
        print(test, "was skipped in every job")

if fail:
    sys.exit(1)
