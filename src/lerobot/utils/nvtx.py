#!/usr/bin/env python

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(message: str):
    pushed = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        try:
            torch.cuda.nvtx.range_push(message)
            pushed = True
        except Exception:
            pushed = False

    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
