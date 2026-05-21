"""Classifier-ready preprocessing layer for the human-vs-AI text dataset."""

from .dataset import Dataset, TextSample, load_splits

__all__ = ["Dataset", "TextSample", "load_splits"]
