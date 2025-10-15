# ---
# jupyter:
#   jupytext:
#     formats: ipynb,python//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: nccs-transport
#     language: python
#     name: python3
# ---

# %%

from utils import open_parcels_output

# %%
ds = open_parcels_output("/mnt/d/nccs-transport/tracks/altimeter/bwd_release_altimeter_nhl_1999.zarr/")

# %%
ds.dropna(dim="trajectory", how="all")
