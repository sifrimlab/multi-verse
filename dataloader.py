import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
import mudatasets


def func1(filename):
    mu.set_options(display_style = "html", display_html_expand = 0b000)

    mdata = mudatasets.load(filename, full=True)
    mdata.var_names_make_unique()

    return mdata
