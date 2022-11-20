from . import bitmask_t, LIBNUMA


def get_bitset_list(bitmask):
    return list(filter(lambda node: LIBNUMA.numa_bitmask_isbitset(bitmask, node) != 0, range(bitmask.contents.size)))
