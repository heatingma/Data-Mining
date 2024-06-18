import os
import requests
import numpy as np
from bs4 import BeautifulSoup


USERAGENT = {
    "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}






class Fetcher():
    def __init__(
        self,
        cache_dir: str = "cache",
        platform: str="windows"
    ) -> None:
        # cache_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # video_id html
        self.video_ids_html_path = os.path.join(cache_dir, "video_ids.html")
        self.video_ids_npy_path = os.path.join(cache_dir, "video_ids.npy")
        
        # platform
        self.platform = platform
    
    def fetch_video_ids(self):
        self.download_view_source(
            url='https://movie.douban.com/top250',
            save_path=self.video_ids_html_path,
            referer='https://movie.douban.com/top250'
        )
    
    def get_video_ids(self) -> np.ndarray:
        with open(self.video_ids_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        import pdb
        pdb.set_trace()
        pass
    
    def download_view_source(
        self, url: str, save_path: str, referer: str,
    ):
        headers = {
            "User-Agent": USERAGENT[self.platform],
            "referer": referer
        }
        res = requests.get(url, headers=headers)
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(res.text)
            
    def fetch(self):
        # fetch video ids
        if not os.path.exists(self.video_ids_npy_path):
            if not os.path.exists(self.video_ids_html_path):
                self.fetch_video_ids()        
            self.get_video_ids()