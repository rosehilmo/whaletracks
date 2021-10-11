"""
Example of getting a detection dataframe.
Usage:
  At command line: 
     source setup_run.sh
     python examples/access/read_detections.py 
"""

import whaletracks.common.constants as cn
from whaletracks.build_data.db_accessor import DBAccessor

accessor = DBAccessor()
df = accessor.df_detection
print(df)
