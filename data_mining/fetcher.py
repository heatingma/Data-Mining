import os
import re
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup


USERAGENT = {
    "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


class Fetcher:
    def __init__(
        self,
        cache_dir: str = "cache",
        platform: str="windows"
    ) -> None:
        # cache_dir and make dirs
        self.cache_dir = cache_dir
        self.movie_ids_dir = os.path.join(cache_dir, "movie_ids")
        self.movie_comments_path = os.path.join(cache_dir, "movie_comments.npy")
        self.movie_comments_html_dir = os.path.join(cache_dir, "movie_comments_html")
        if not os.path.exists(self.movie_ids_dir):
            os.makedirs(self.movie_ids_dir)
        if not os.path.exists(self.movie_comments_html_dir):
            os.makedirs(self.movie_comments_html_dir)
              
        # movie_id html
        self.movie_ids_html_base_path = os.path.join(self.movie_ids_dir, "movie_ids")
        self.movie_ids_names_npy_path = os.path.join(self.movie_ids_dir, "movie_ids_names.npy")
        
        # platform
        self.platform = platform    
    
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
    
    def fetch_movie_ids(self):
        for page_idx in range(10):
            start_idx = page_idx * 25
            url = f'https://movie.douban.com/top250?start={start_idx}&filter='
            save_path = self.movie_ids_html_base_path + f"-{page_idx}.html"
            if not os.path.exists(save_path):
                self.download_view_source(
                    url=url, save_path=save_path, referer='https://movie.douban.com/top250'
                )
    
    def get_movie_ids(self):
        movie_ids = list()
        movies_names = list()
        for page_idx in range(10):
            save_path = self.movie_ids_html_base_path + f"-{page_idx}.html"
            with open(save_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            target_0 = soup.find_all(class_="pic")
            target_1 = [_target_0.find("a")['href'] for _target_0 in target_0]
            target_2 = [int(_target_1.split("/")[-2]) for _target_1 in target_1]
            names = [_target_0.find('img')["alt"] for _target_0 in target_0]
            movie_ids += target_2
            movies_names += names
        movie_ids = np.array(movie_ids)
        movies_names = np.array(movies_names)    
        movies_ids_names = dict()
        for idx in range(len(movie_ids)):
            movies_ids_names[movie_ids[idx]] = movies_names[idx]
        self.movies_ids_names = movies_ids_names
        np.save(self.movie_ids_names_npy_path, self.movies_ids_names)

    def fetch_movie_comments_by_id(self, movie_id: int):
        # save dir
        save_dir = os.path.join(self.movie_comments_html_dir, str(movie_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fetch
        for page_idx in range(10):
            start_idx = page_idx * 20
            url = f"https://movie.douban.com/subject/{movie_id}/comments?start={start_idx}&limit=20&status=P&sort=new_score"
            save_path = os.path.join(save_dir, f"{movie_id}_{start_idx}_{start_idx+19}.html")
            if not os.path.exists(save_path):
                self.download_view_source(
                    url=url, save_path=save_path, referer='https://movie.douban.com/'
                )
    
    def fetch_movie_comments(self):
        for movie_id in tqdm(self.movies_ids_names.keys()):
            self.fetch_movie_comments_by_id(movie_id)
    
    def extract_movie_comments_by_id(self, movie_id: int) -> list:
        source_dir = os.path.join(self.movie_comments_html_dir, str(movie_id))
        comments = list()
        html_files_names = os.listdir(source_dir)
        for html_files_name in html_files_names:
            html_files_path = os.path.join(source_dir, html_files_name)
            with open(html_files_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')    
            target_0 = soup.find_all(class_="comment")
            soup.find_all(class_="comment-report")
            for _target_0 in target_0:
                comment = _target_0.find('span', 'short').text
                like = int(_target_0.find('span', 'votes vote-count').text)
                star = _target_0.find('span', {'class': re.compile(r'allstar\d+ rating')})
                try:
                    star = int(star['class'][0][-2])
                except:
                    star = -1
                time = _target_0.find('span', 'comment-time').text.split('\n')[1].strip()
                comments.append([movie_id, self.movies_ids_names[movie_id], comment, like, star, time])
        return comments
                
    def extract_movie_comments(self):
        if not os.path.exists(self.movie_comments_path):
            all_comments = list()
            for movie_id in tqdm(self.movies_ids_names.keys()):
                comments = self.extract_movie_comments_by_id(movie_id)
                all_comments += comments
            self.all_comments = np.array(all_comments)
            np.save(self.movie_comments_path, self.all_comments)
        else:
            self.all_comments = np.load(self.movie_comments_path, allow_pickle=True)
            
    def fetch(self):
        # fetch movie ids
        if not os.path.exists(self.movie_ids_names_npy_path):
            self.fetch_movie_ids()        
            self.get_movie_ids()
        else:
            self.movies_ids_names = np.load(
                self.movie_ids_names_npy_path, allow_pickle=True
            ).item()
            
        # fetch comments
        self.fetch_movie_comments()
        self.extract_movie_comments()

        import pdb
        pdb.set_trace()