import re

def find_bb_size(line):
    bb_size = 2; # default value
    m = re.search(r'\(\$(\d+\.?\d*)/\$(\d+\.?\d*)\)', line)
    if m:
        bb_size = float(m.group(2))
    else:
        print("ERROR!!! something went wrong in finding the bb size")
    return bb_size

