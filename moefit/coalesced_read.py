"""Bounded within-batch extent coalescing on an unchanged immutable file.
Caller owns fd and executor. No speculative reads across batches/layers.
Returned memoryviews preserve input order and keep their backing buffers alive.
"""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Span:
    offset: int
    length: int
    members: tuple  # (caller index, relative offset, length)


def plan_reads(extents, *, max_gap=0, max_span=1024*1024, max_amplification=1.125):
    if max_gap < 0 or max_span <= 0 or max_amplification < 1:
        raise ValueError('invalid coalescing bounds')
    items = []
    for i, (off, size) in enumerate(extents):
        if not isinstance(off, int) or not isinstance(size, int) or off < 0 or size <= 0:
            raise ValueError('extents require nonnegative integer offsets and positive sizes')
        if size > max_span:
            raise ValueError('single extent exceeds max_span')
        items.append((off, off+size, i))
    items.sort()
    spans = []
    start = end = useful = 0
    members = []
    for off, stop, i in items:
        added = max(0, stop-max(end, off)) if members else stop-off
        merged_end = max(end, stop)
        can_merge = (members and off-end <= max_gap and merged_end-start <= max_span
                     and merged_end-start <= max_amplification*(useful+added))
        if members and not can_merge:
            spans.append(Span(start, end-start, tuple(members)))
            members = []
        if not members:
            start, end, useful = off, stop, stop-off
        else:
            end, useful = merged_end, useful+added
        members.append((i, off-start, stop-off))
    if members:
        spans.append(Span(start, end-start, tuple(members)))
    return spans


def read_exact(fd, size, offset):
    parts = []
    remaining = size
    while remaining:
        block = os.pread(fd, remaining, offset)
        if not block:
            raise EOFError(f'short read at {offset}, missing {remaining} bytes')
        parts.append(block)
        offset += len(block)
        remaining -= len(block)
    return parts[0] if len(parts) == 1 else b''.join(parts)


def read_batch(fd, extents, *, executor=None, lanes=0, **bounds):
    extents = list(extents)
    spans = plan_reads(extents, **bounds)
    def fetch(span):
        return read_exact(fd, span.length, span.offset)
    if lanes < 0 or not isinstance(lanes, int):
        raise ValueError('lanes must be a nonnegative integer')
    if executor is not None and lanes and spans:
        # One future per sequential lane, not per tiny extent. Contiguous
        # sorted partitions preserve ascending reads within each worker.
        width = (len(spans) + lanes - 1) // lanes
        chunks = [spans[i:i+width] for i in range(0, len(spans), width)]
        def fetch_lane(chunk):
            return [(span, fetch(span)) for span in chunk]
        pairs = (pair for group in executor.map(fetch_lane, chunks) for pair in group)
    else:
        blocks = map(fetch, spans) if executor is None else executor.map(fetch, spans)
        pairs = zip(spans, blocks)
    out: list[memoryview | None] = [None] * len(extents)
    for span, block in pairs:
        view = memoryview(block)
        for index, offset, size in span.members:
            out[index] = view[offset:offset+size]
    assert all(x is not None for x in out)
    return [x for x in out if x is not None], {'planned_reads': len(spans), 'issued_bytes': sum(s.length for s in spans),
                 'requested_bytes': sum(size for _, size in extents)}
