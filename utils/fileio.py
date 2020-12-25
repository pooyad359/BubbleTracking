import sys
import os
import urllib
import threading
from queue import Queue
import requests
from tqdm.auto import tqdm
from pathlib import Path

class DownloadThread(threading.Thread):
    def __init__(self, queue, destfolder):
        super(DownloadThread, self).__init__()
        self.queue = queue
        self.destfolder = destfolder
        self.daemon = True

    def run(self):
        while True:
            filename, url = self.queue.get()
            try:
                self.download_url(url,filename)
            except Exception as e:
                print(f"   Error:{e}")
            self.queue.task_done()

    def download_url(self, url,filename):
        # change it to a different way if you require
        name = url.split('/')[-1]
        path = os.path.join(self.destfolder, filename)
        print(f"[{self.ident}] Downloading {url} -> {filename}")
        resp = requests.get(url)
        if resp.status_code == 200:
            content = resp.content
            with open(path,'wb') as fp:
                fp.write(content)
        else:
            print(f'Unable to download {url}')
#         urllib.request.urlretrieve(url, dest)

def download(urls, destfolder, numthreads=4):
    queue = Queue()
    path = Path(destfolder)
    path.mkdir(parents=True,exist_ok = True)
    for filename, url in urls.items():
        queue.put((filename, url))

    for i in range(numthreads):
        t = DownloadThread(queue, destfolder)
        t.start()

    queue.join()

