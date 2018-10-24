import sys


def load_data(filename, separator='\t'):
    _src_list = []
    _target_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                src, dest = line.strip().split(separator)
                _src_list.append(src)
                _target_list.append(dest)
            except ValueError:
                print("File format exception : ", line, file=sys.stderr)
    return _src_list, _target_list
