"""Access methods for acquiring data from the Whale Database"""

from common_python.database import database_util as util
import whaletracks.common.constants as cn
from whaletracks.common import util

from obspy.core.utcdatetime import UTCDateTime
import os
import pandas as pd
import numpy as np
import sqlite3


#Tables
TABLE_STATIONS = "stations"
TABLE_DETECTIONS = "detections"
TABLE_CHANNELS = "channels"


class DBAccessor(object):

  def __init__(self, db_pth=cn.DB_PTH):
    """
    :param str network_code:
    :param UTCDateTime start_time: starting time of data
    :param UTCDateTime end_time: starting time of data
    """
    self._conn = None
    self._db_pth = db_pth
    self._tables = {}  # Dataframes keyed by table name

  @property
  def df_station(self):
    return self._readTable(cn.SCM_STATION.tablename)

  @property
  def df_channel(self):
    df = self._readTable(cn.SCM_CHANNEL.tablename)
    # Convert datetimes back to string
    for col in [cn.START_TIME, cn.END_TIME]:
      df[col] = [UTCDateTime(t) for t in df[col]]
    for col in [cn.POLES, cn.ZEROES]:
      complexes = []
      for stg in df[col]:
        cmplx = util.complexifyString(stg)
        complexes.append(cmplx)
      df[col] = complexes
    return df

  @property
  def df_detection(self):
    return self._readTable(cn.SCM_DETECTION.tablename)

  def _connect(self):
    self._conn = sqlite3.connect(self._db_pth)

  def _close(self):
    if self._conn is not None:
      self._conn.close()
      self._conn = None

  def _readTable(self, tablename):
    """
    Reads the specified table from the whales database.
    :param str tablename:
    :return pd.DataFrame:
    """
    if not tablename in self._tables.keys():
      self._connect()
      query = "select * from %s" % tablename
      self._tables[tablename] = pd.read_sql_query(query, self._conn)
      self._close()
    return self._tables[tablename]
