# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
output2 = dataiku.Dataset("output2")
output2_df = output2.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

output_test_df = output2_df # For this sample code, simply copy input to output


# Write recipe outputs
output_test = dataiku.Dataset("output_test")
output_test.write_with_schema(output_test_df)
