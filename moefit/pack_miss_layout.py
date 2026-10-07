"""Cache-aware training batches for immutable expert layout, not eviction tuning."""
from collections import OrderedDict
from moefit.pack_layout import co_demand_order


def miss_batches(routes, capacity):
    """One layer's routes, empty initial LRU; current hits protected first.

    Returns misses in original route order. No heldout data belongs here when
    fitting a layout. Residency policy is fixed independently of physical order.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError('capacity must be positive integer')
    cache = OrderedDict()
    for batch in routes:
        ids = list(dict.fromkeys(batch))
        if len(ids) > capacity:
            raise ValueError('capacity must hold current route')
        misses = [e for e in ids if e not in cache]
        for e in ids:
            if e in cache: cache.move_to_end(e)
        for e in misses:
            if len(cache) == capacity: cache.popitem(last=False)
            cache[e] = None
        yield misses


def co_miss_order(routes, experts, capacity):
    return co_demand_order(miss_batches(routes, capacity), experts)
