from . import LIBNUMA
from . import utils as numa_utils

__all__ = ["numa_available", "get_max_node", "get_max_possible_node", "get_num_configured_nodes",
           "get_num_configured_cpus", "numa_distance", "numa_hardware_info"]


def numa_available():
    return LIBNUMA.numa_available() != -1


def get_max_node():
    return LIBNUMA.numa_max_node()


def get_max_possible_node():
    return LIBNUMA.numa_max_possible_node()


def get_num_configured_nodes():
    return LIBNUMA.numa_num_configured_nodes()


def get_num_configured_cpus():
    return LIBNUMA.numa_num_configured_cpus()


def numa_distance(node1, node2):
    return LIBNUMA.numa_distance(node1, node2)


def numa_hardware_info():
    """
    :return: Dict(numa_node_distance:List[List[int]], node_cpu_info:Dict(node:List[int]))
    """
    # handle numa node distance
    numa_node_distance = []
    for i in range(get_num_configured_nodes()):
        tmp_distance = []
        for j in range(get_num_configured_nodes()):
            tmp_distance.append(numa_distance(i, j))
        numa_node_distance.append(tmp_distance)

    # handle cpu info
    node_cpu_info = {}
    for i in range(get_num_configured_nodes()):
        node_cpu_info[i] = node_to_cpus(i)

    return {"numa_node_distance": numa_node_distance, "node_cpu_info": node_cpu_info}


def cpu_to_node(cpu):
    return LIBNUMA.numa_node_of_cpu(cpu)


def node_to_cpus(node):
    cpu_mask = LIBNUMA.numa_allocate_cpumask()
    LIBNUMA.numa_bitmask_clearall(cpu_mask)
    res = LIBNUMA.numa_node_to_cpus(node, cpu_mask)
    if res == 0:
        return numa_utils.get_bitset_list(cpu_mask)
    else:
        return []
