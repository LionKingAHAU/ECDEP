"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-05-30
In this part, we use eTILES to extract communities emerged during the DPPIN.
"""

import tiles as t
print("---------------STEP3---------------")
# You can change the path to your own configs
inPath = "../data/Dynamic Network Demo/streaming source.tsv"
t1 = t.eTILES(inPath, path="result", obs=3)
t1.execute()
