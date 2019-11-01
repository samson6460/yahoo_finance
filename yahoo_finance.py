import requests                  # [handles the http interactions](http://docs.python-requests.org/en/master/) 
from bs4 import BeautifulSoup    # beautiful soup handles the html to text conversion and more
import re                        # regular expressions are necessary for finding the crumb (more on crumbs later)
from time import mktime          # mktime transforms datetime objects to unix timestamps
import numpy as np
from matplotlib.dates import date2num
import datetime as dtt
from urllib.error import HTTPError

stock_dt_ohlc = np.dtype([
    (str('date'), object),
    (str('year'), np.int16),
    (str('month'), np.int8),
    (str('day'), np.int8),
    (str('d'), float),     # mpl datenum
    (str('open'), float),
    (str('high'), float),
    (str('low'), float),
    (str('close'), float),
    (str('volume'), float),
    (str('aclose'), float)])


stock_dt_ochl = np.dtype(
    [(str('date'), object),
     (str('year'), np.int16),
     (str('month'), np.int8),
     (str('day'), np.int8),
     (str('d'), float),     # mpl datenum
     (str('open'), float),
     (str('close'), float),
     (str('high'), float),
     (str('low'), float),
     (str('volume'), float),
     (str('aclose'), float)])

def _get_crumbs_and_cookies(stock):
    """
    get crumb and cookies for historical data csv download from yahoo finance
    
    Parameters
    ----------
    stock : str
        short-handle identifier of the company 
    
    If successful, returns a tuple of header, crumb and cookie
    or a tuple of None
    """
    
    url = 'https://finance.yahoo.com/quote/{}/history'.format(stock)
    with requests.session():
        header = {'Connection': 'keep-alive',
                   'Expires': '-1',
                   'Upgrade-Insecure-Requests': '1',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                   AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                   }
        
        website = requests.get(url, headers=header)
        soup = BeautifulSoup(website.text, 'lxml')
        crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))
        if crumb == []:
            return (None, None, None)
        return (header, crumb[0], website.cookies)
    
def convert_to_unix(date):
    """
    converts datetime to unix timestamp
    
    returns integer unix timestamp
    """    
    return int(mktime(date.timetuple()))

def load_csv_data(stock, day_begin, day_end, rep_time=5, interval='1d'):
    """
    queries yahoo finance api to receive historical data in csv file format
    
    Parameters
    ----------
        stock : str
            short-handle identifier of the company               
        
        day_begin : datetime
            starting date for the historical data
        
        day_end : datetime
            final date of the data
        
        rep_time : int 
            Sometimes it will fail when getting the crum or historical data.
            Assign the parm to set the number of repetitions to get crum 
            and cookies. The default is 5.
        
        interval : str
            1d (default) , 1wk, 1mo - daily, weekly monthly data
    
    returns a list of comma seperated value lines
    """
    day_begin_unix = convert_to_unix(day_begin)
    day_end_unix = convert_to_unix(day_end)
    
    header, crumb, cookies = _get_crumbs_and_cookies(stock)
    
    with requests.session():
        for _ in range(rep_time):
            url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
                  '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events=history&crumb={crumb}' \
                  .format(stock=stock, day_begin=day_begin_unix, day_end=day_end_unix, interval=interval, crumb=crumb)
                    
            website = requests.get(url, headers=header, cookies=cookies)
            if "Unauthorized" not in website.text and crumb is not None:
                break
            header, crumb, cookies = _get_crumbs_and_cookies(stock)
        else:
            if crumb is None:
                raise ValueError('could not find the stock:' + stock)
            else:
                raise HTTPError(url, website, "Unauthorized", header, 1)
        return website.text.split('\n')[:-1]
    
def quotes_historical_yahoo_ochl(stock, date1, date2,
                                 adjusted=True, asobject=False,
                                 ochl=True, rep_time=5):

    """Parse the historical data in file handle fh from yahoo finance.


    Parameters
    ----------
        stock : str
            short-handle identifier of the company  
                
        date1 : datetime
            starting date for the historical data
            
        date2 : datetime
            final date of the data
    
        adjusted : bool
            If True (default) replace open, high, low, close prices with
            their adjusted values. The adjustment is by a scale factor, S =
            adjusted_close/close. Adjusted prices are actual prices
            multiplied by S.
            
            Volume is not adjusted as it is already backward split adjusted
            by Yahoo. If you want to compute dollars traded, multiply volume
            by the adjusted close, regardless of whether you choose adjusted
            = True|False.
    
    
        asobject : bool or None
            If False (default for compatibility with earlier versions)
            return a list of tuples containing
            
                d, open, high, low, close, volume
            
            or
            
                d, open, close, high, low, volume
            
            depending on `ochl`
            
            If None (preferred alternative to False), return
            a 2-D ndarray corresponding to the list of tuples.
            
            Otherwise return a numpy recarray with
            
                date, year, month, day, d, open, high, low, close,
                volume, adjusted_close
            
            where d is a floating poing representation of date,
            as returned by date2num, and date is a python standard
            library datetime.date instance.
            
            The name of this kwarg is a historical artifact.  Formerly,
            True returned a cbook Bunch
            holding 1-D ndarrays.  The behavior of a numpy recarray is
            very similar to the Bunch.
    
        ochl : bool
            Selects between ochl and ohlc ordering.
            Defaults to True to preserve original functionality.
            
        rep_time : int 
            Sometimes it will fail when getting the crum or historical data.
            Assign the parm to set the number of repetitions to get crum 
            and cookies. The default is 5.

    """
    
    fh = load_csv_data(stock, day_begin=date1, day_end=date2, rep_time=5, interval='1d')
    
    if ochl:
        stock_dt = stock_dt_ochl
    else:
        stock_dt = stock_dt_ohlc

    results = []

    #    datefmt = '%Y-%m-%d'
    #fh.readline()  # discard heading
    for line in fh[1:]:

        vals = line.split(',')
        if len(vals) != 7:
            continue      # add warning?
        datestr = vals[0]
        #dt = datetime.date(*time.strptime(datestr, datefmt)[:3])
        # Using strptime doubles the runtime. With the present
        # format, we don't need it.
        dt = dtt.date(*[int(val) for val in datestr.split('-')])
        dnum = date2num(dt)
        open, high, low, close = [0 if val=='null' else float(val) for val in vals[1:5]]
        aclose = 0 if vals[5]=='null' else float(vals[5])
        volume = 0 if vals[6]=='null' else float(vals[6])
        if ochl:
            results.append((dt, dt.year, dt.month, dt.day,
                            dnum, open, close, high, low, volume, aclose))

        else:
            results.append((dt, dt.year, dt.month, dt.day,
                            dnum, open, high, low, close, volume, aclose))
    results.reverse()
    d = np.array(results, dtype=stock_dt)
    if adjusted:
        scale = d['aclose'] / d['close']
        scale[np.isinf(scale)] = np.nan
        d['open'] *= scale
        d['high'] *= scale
        d['low'] *= scale
        d['close'] *= scale

    if not asobject:
        # 2-D sequence; formerly list of tuples, now ndarray
        ret = np.zeros((len(d), 6), dtype=float)
        ret[:, 0] = d['d']
        if ochl:
            ret[:, 1] = d['open']
            ret[:, 2] = d['close']
            ret[:, 3] = d['high']
            ret[:, 4] = d['low']
        else:
            ret[:, 1] = d['open']
            ret[:, 2] = d['high']
            ret[:, 3] = d['low']
            ret[:, 4] = d['close']
        ret[:, 5] = d['volume']
        if asobject is None:
            return ret
        return [tuple(row) for row in ret]
    
    return d.view(np.recarray)  # Close enough to former Bunch return