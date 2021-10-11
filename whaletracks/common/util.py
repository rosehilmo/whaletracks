"""Utilities for Whale Project."""

from datetime import datetime


def complexifyString(stg, separator=";"):
  """
  Converts a string representation of complex numbers into
  a list of complex numbers.
  :param str stg: string of complex numbers
  """
  splits = stg.split(separator)
  if len(splits[0]) > 0:
    return [complex(eval(s)[0], eval(s)[1]) for s in splits]
  else:
    return []




def datestrToEpoch(datestrs,dateformat='%Y-%m-%dT%H:%M:%S.%fZ'):
    """
    Converts list of datestrings in UTCDateTime format
    into epoch time (seconds elapsed since 01-01-1970)
    :param list datestrs: list of datestrings
    :param str dateformat: date format as accepted by datetime.strptime
    :return list floats: seconds elapsed since 01-01-1970
    """
    #import pdb; pdb.set_trace()
    epochlist=list()
    SECONDS_IN_DAY=86400
    datetime_epochstart = datetime.strptime('1970-01-01T00:00:00.00Z',
                                            '%Y-%m-%dT%H:%M:%S.%fZ')
    for k in range(1,len(datestrs)+1):
        j=k-1
        datetime_j = datetime.strptime(datestrs[j],dateformat)
        epoch_delta = datetime_j - datetime_epochstart
        epoch_j = epoch_delta.days*SECONDS_IN_DAY + epoch_delta.seconds + epoch_delta.microseconds/1000000
        
        epochlist.append(epoch_j)
        
    return epochlist

def datetimeToEpoch(UTCdatetime_list):
    """
    Converts list of datestrings in UTCDateTime format
    into epoch time (seconds elapsed since 01-01-1970)
    :param list datestrs: list of datestrings
    :param str dateformat: date format as accepted by datetime.strptime
    :return list floats: seconds elapsed since 01-01-1970
    """
    #import pdb; pdb.set_trace()
    epochlist=list()
    SECONDS_IN_DAY=86400
    datetime_epochstart = datetime.strptime('1970-01-01T00:00:00.00Z',
                                            '%Y-%m-%dT%H:%M:%S.%fZ')

    #import pdb; pdb.set_trace()
    for k in range(1,len(UTCdatetime_list)+1):
        j=k-1
        datetime_j = UTCdatetime_list[j]
        epoch_delta = datetime_j.datetime - datetime_epochstart
        epoch_j = epoch_delta.days*SECONDS_IN_DAY + epoch_delta.seconds + epoch_delta.microseconds/1000000
        
        epochlist.append(epoch_j)
        
    return epochlist


def addEpochColumns(dataframe):
    ENDING_STRING = "_TIME"
    for col in dataframe.columns:
        if ENDING_STRING in col:
            front = col[0:len(col)-len(ENDING_STRING)]
            new_col = "%s_EPOCH" % front
            dataframe[new_col] = datestrToEpoch(dataframe[col])
            
    return dataframe
    
    
    
    
        
    
    
    
    