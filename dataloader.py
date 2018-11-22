import sys


def load_data(filename, separator='\t'):
    _src_list = []
    _target_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if len(line.strip()) == 0:
                    continue
                src, dest = line.strip().split(separator)
                _src_list.append(src)
                _target_list.append(dest)
                # if len(_target_list) > 5000:
                #     break
            except ValueError:
                print("File format exception : ", line, file=sys.stderr)
    return _src_list, _target_list
