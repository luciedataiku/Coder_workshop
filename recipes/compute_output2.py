# -*- coding: utf-8 -*-
import dataiku

# Read recipe inputs
web_data_cloud_storage = dataiku.Dataset("web_data_cloud_storage")
web_data_cloud_storage_df = web_data_cloud_storage.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

output2_df = web_data_cloud_storage_df # For this sample code, simply copy input to output


# Write recipe outputs
output2 = dataiku.Dataset("output2")
output2.write_with_schema(output2_df)
