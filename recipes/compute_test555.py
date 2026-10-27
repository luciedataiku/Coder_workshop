# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
unlabeled_scored_autoML = dataiku.Dataset("unlabeled_scored_autoML")
unlabeled_scored_autoML_df = unlabeled_scored_autoML.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

test555_df = unlabeled_scored_autoML_df # For this sample code, simply copy input to output


# Write recipe outputs
test555 = dataiku.Dataset("test555")
test555.write_with_schema(test555_df)
