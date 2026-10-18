"""Wrap wpa_pipeline.py into a single-cell Jupyter notebook."""
import json, io

src = io.open("wpa_pipeline.py", encoding="utf-8").read()

# The notebook version renders figures inline.
src = src.replace(
    "# ----------------------------- 1. IMPORTS ------------------------------------",
    "# ----------------------------- 1. IMPORTS ------------------------------------\n"
    "try:\n"
    "    get_ipython().run_line_magic('matplotlib', 'inline')\n"
    "except Exception:\n"
    "    pass\n", 1)

nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with io.open("WPA_IPL_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote WPA_IPL_Pipeline.ipynb  (1 cell,", len(src.splitlines()), "lines )")
