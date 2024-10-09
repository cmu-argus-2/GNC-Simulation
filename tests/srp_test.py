import sys
sys.path.append('../')

from build.tests.pymodels import sun_pos

import pytest
import numpy as np
import requests
import re
import datetime
from tqdm import tqdm
import os

def parse_html(text):
    data_start = text.find("<tbody>")
    data_end = text.find("</tbody>")
    result_str = text[data_start+7:data_end]
    res_strings = re.findall("<td>.+[0-9]</td>", result_str)

    pos = [float(p[4:-5]) for p in res_strings]
    return pos

def utc2tJ2000(Y, M, D):
    a = int(Y / 100.0)
    b = 2 - a + int(a / 4.0)

    jd=int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + D + b - 1524.5
    tJ2000 = (jd-2451544.5)*24*60*60

    return tJ2000


# generate truth data
# Sends a POST request, parses the html result and stores positions in a csv file
if not os.path.isfile("sol_pos_ECI.csv"):
    url = "https://astroconverter.com/sunpos.html"

    session = requests.Session()
    dates = [datetime.datetime.today() - datetime.timedelta(days=x-800) for x in range(1000)]
    datelist = [d.strftime("%Y-%m-%d") + "T00:00:00" for d in dates]

    truth = np.zeros((len(datelist), 4))
    for i in tqdm(range(len(datelist))):
        payload = {'utc' : datelist[i]}
        r = session.post(url, payload)
        truth[i, 1:] = parse_html(r.text)
        truth[i,0] = utc2tJ2000(dates[i].year, dates[i].month, dates[i].day)

    np.savetxt('sol_pos_ECI.csv', truth, delimiter=',')

truth = np.genfromtxt('sol_pos_ECI.csv', delimiter = ',')

test_cases = [(truth[i,0], list(1000*truth[i,1:])) for i in range(len(truth))]
@pytest.mark.parametrize("date, truth", test_cases[0:10])
def test(date, truth):
    pos = np.array(sun_pos(date))

    assert np.linalg.norm(pos-truth) <= 100 # field error against ground truth <= 10nT


