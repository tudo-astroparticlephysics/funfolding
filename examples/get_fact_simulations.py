import os
import requests


URL = 'http://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'

script_dir = os.path.dirname(os.path.abspath(__file__))


def download(url=URL):
    path = os.path.join(script_dir, "fact_simulations.hdf")
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)

