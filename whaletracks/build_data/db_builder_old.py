"""Acquires data used from obspy and builds the database."""

import sys
sys.path.append('/Users/wader/Desktop/whaletracks/') 
sys.path.append('/Users/wader/Desktop/whaletracks/common_python') 

from common_python.database import database_util as util
import whaletracks.common. constants as cn

import copy
import matplotlib.pyplot as plt
import obspy
import os
from obspy.clients.fdsn import Client
from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
import numpy as np


UNNAMED_COLUMN = "Unnamed:"  # Bogus column added
CREATION_DATE = "creation_date"
END_DATE = "end_date"
EPOCH = "_epoch"
START_DATE = "start_date"
TERMINATION_DATE = "termination_date"
STATION_SWAPS = [
    (CREATION_DATE, cn.CREATION_TIME),
    (END_DATE, cn.END_TIME),
    (START_DATE, cn.START_TIME),
    (TERMINATION_DATE, cn.TERMINATION_TIME),
    ]

def fillEpochColumns(df):
  """
  Temporary code that populates the _EPOCH columns with None
  :param pd.DataFrame df: some columns have "_time"
  """
  ENDING = "_time"
  for col in df.columns:
    if df[col] is not None:
      if len(df[col]) > 0:
        length = len(df[col])
    if EPOCH in col:
      df[col] = list(np.repeat(None, length))
    if ENDING in col:
      new_col = col[0:len(col)-len(ENDING)]
      new_col = "%s_epoch" % new_col
      df[new_col] = list(np.repeat(None, length))

################### HELPER FUNCTIONS ########################
def _makeChannelID(network_code, station_code, channel_code):
  return "%s.%s.%s" % (network_code, station_code, channel_code)

def _makeStationID(network_code, station_code):
  return "%s.%s" % (network_code, station_code)

def _stringifyComplexs(complexs):
  return ";".join([str((x.real, x.imag)) for x in complexs])

def _getPolesZeroes(channel):
  try:
      zeroes = _stringifyComplexs(channel.response.get_paz().zeros)
  except:
      zeroes = ""
  try:
      poles = _stringifyComplexs(channel.response.get_paz().poles)
  except:
      poles = ""
  return poles, zeroes


################### CLASS ########################
class DBBuilder(object):
  """
  Builds the whaletracks database. Useage:
    builder = DBBuilder()
    builder.build()
  Database path is specified in constants.py as DB_PTH.
  """

  def __init__(self, network_code="7D", channel="*Z",
      start_time= UTCDateTime("2001-01-01"), 
      end_time=UTCDateTime("2020-01-02"), db_pth=cn.DB_PTH):
    """
    :param str network_code:
    :param UTCDateTime start_time: starting time of data
    :param UTCDateTime end_time: starting time of data
    Usage
      builder = DBUilder()
      builder.build()
    """
    self._network_code = network_code
    self._start_time = start_time
    self._end_time = end_time
    self._db_pth = db_pth
    #
    self.client = Client("IRIS")
    self.inventory = self.client.get_stations(
        network=self._network_code,
        station="*", channel=channel, level="response", 
        starttime=self._start_time, endtime=self._end_time)
    self.network = self.inventory[0]

  def build(self):
    """
    Builds all of the database tables.
    """
    # Create the dataframes and associate with table names
    df_dct = {}
    df_dct[cn.SCM_STATION.tablename] = self._makeStationDF()
    #
    df_channel = self._makeChannelDF()
    df_channel[cn.END_TIME] = [str(d) for d in df_channel[cn.END_TIME]]
    df_channel[cn.START_TIME] = [str(d) for d in df_channel[cn.START_TIME]]
    df_dct[cn.SCM_CHANNEL.tablename] = df_channel
    # Create tables for schemas with csv files
    for schema in cn.SCMS:
      if schema.csv_path is not None:
        df_dct[schema.tablename] = DBBuilder._readCSV(schema.csv_path)
    # Add epoch columns, days since 1970 for all columns ending in _TIME
    #for df in df_dct.values():
      #util.addEpochColumns(df)
      # Temporary code to fill in EPOCH columns
      #fillEpochColumns(df)
    # Write the tables
    for tablename, df in df_dct.items():
      util.updateDBTable(df, self._db_pth, tablename)
    
                           
  ########## Table building methods. ##############
  def _makeStationDF(self):
    """
    Constructs a dataframe of station metadata.
    """
    CODE = "code"
    attributes = list(cn.SCM_STATION.columns)
    attributes.remove(cn.NETWORK_CODE)
    attributes.remove(cn.STATION_CODE)
    attributes.append(CODE)
    # Old key names are used in the data
    for key_old, key_new in STATION_SWAPS:
      attributes.remove(key_new)
      attributes.append(key_old)
    # Remove EPOCH columns since they are generated
    for col in list(attributes):
      if EPOCH in col:
        attributes.remove(col)
    #
    station_count = self.network.total_number_of_stations
    station_dct = {k: [] for k in attributes}
    for station in self.network:
      for key in attributes:
        station_dct[key].append(str(station.__getattribute__(key)))
    station_dct[cn.STATION_CODE] = station_dct[CODE]
    station_dct[cn.NETWORK_CODE] = np.repeat(self._network_code,
        len(station_dct[cn.STATION_CODE]))
    del station_dct[CODE]
    for key_old, key_new in STATION_SWAPS:
      station_dct[key_new] = station_dct[key_old]
      del station_dct[key_old]
    #fillEpochColumns(station_dct)
    return pd.DataFrame(station_dct)

  @staticmethod
  def _readCSV(path):
    """
    Constructs a dataframe from a CSV file.
    Some cleanup is done, like removing "Unnamed:" colums.
    :param str path:
    """
    df = pd.read_csv(path)
    for col in df.columns:
      if UNNAMED_COLUMN in col:
        del df[col]
    return df
        
  def _makeChannelDF(self):
    """
    :param obspy.inventory.network.Network network:
    :return pd.DataFrame: channels, poles, zeros
      channel dataframe: cn.SCM_CHANNEL.columns
      poles, zeros
          
    """
    channel_dct = {k: [] for k in cn.SCM_CHANNEL.columns}
    # Remove generated columns
    keys = list(channel_dct.keys())
    for key in keys:
      if EPOCH in key:
        del channel_dct[key]
    #
    for station in self.network:
      station_id = _makeStationID(self.network.code, station.code)
      for channel in station:
        poles, zeroes = _getPolesZeroes(channel)
        channel_dct[cn.AZIMUTH].append(channel.azimuth)
        channel_dct[cn.CHANNEL_ID].append(_makeChannelID(
            self.network.code, station.code, channel.code))
        channel_dct[cn.CHANNEL_TYPES].append(" ".join(channel.types))
        channel_dct[cn.DIP].append(channel.dip)
        channel_dct[cn.END_TIME].append(channel.end_date)
        channel_dct[cn.POLES].append(poles)
        channel_dct[cn.SENSITIVITY_FREQUENCY].append(channel.response.instrument_sensitivity.frequency)
        channel_dct[cn.SENSITIVITY_VALUE].append(channel.response.instrument_sensitivity.value)
        channel_dct[cn.SENSOR].append(channel.sensor.description)
        channel_dct[cn.START_TIME].append(channel.start_date)
        channel_dct[cn.STATION_ID].append(_makeStationID(
            self.network.code, station.code))
        channel_dct[cn.ZEROES].append(zeroes)
    #fillEpochColumns(channel_dct)
    return pd.DataFrame(channel_dct)


if __name__ == '__main__':
  # Create the database
  builder = DBBuilder()
  builder.build()
