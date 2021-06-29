from selenium import webdriver

class Articles:
    def __init__(self):
        self.article_id=0
        self.list={}
    
    def add_article(self, title, keywords, date,
                    datetime, press, url, body):
        self.article_id+=1
        self.list.update({self.article_id :{"title": title,
                             "keywords" : keywords,
                             "date": date,
                             "datetime": datetime,
                             "press":press,
                             "url": url,
                             "body": body}})
        self.save_check()
    
    def save_check(self, savepoint=100):
        if len(self.list) % savepoint == 0:
            import json
            with open("./articles/articles2.json", "w", encoding='UTF-8-sig') as js:
                dmp=self.list.copy()
                js.write(json.dumps(dmp, ensure_ascii=False))
                dmp=None

class Crawler:
    def __init__(self, urls):
        self.new_browser()
        self.articles_loader()
        self.urls = urls
        self.error_urls=set()
        
    def new_browser(self):
        from selenium.webdriver.firefox.options import Options
        options = Options()
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox") # ONLY ON UBUNTU!
        options.add_argument("--headless")
        self.browser = webdriver.Firefox(options=options)
    
    def articles_loader(self):
        import os, json
        if not os.path.exists("./articles/articles2.json"):
            if "articles" not in os.listdir():
                os.mkdir("articles")
            self.checkpoint=0
            self.articles = Articles()
            print("Crawler Initialized")
        else:
            self.articles = Articles()
            self.articles.list = json.load(open("./articles/articles2.json", "r", encoding="UTF-8-sig"))
            self.checkpoint = int(open("./articles/checkpoint2", "r").read())
            self.articles.article_id = self.checkpoint+1

            print("Current checkpoint: {}".format(self.checkpoint))
            print("Current Articles length: {}".format(len(self.articles.list)))
            print("Next Article ID: {}".format(self.articles.article_id))
            print()
            print("Crawler Loaded")
    
    def save_checkpoint(self):
        with open("./articles/checkpoint2", "w") as writer:
            writer.write(str(self.checkpoint))
        self.browser.close()
        self.new_browser()
    
    def crawl_urls(self):
        from tqdm import tqdm
        with tqdm(self.urls, leave=False, bar_format="{percentage:2.2f}%{bar}{desc} 예상남은시간: {remaining}") as t:
            for i, url in enumerate(self.urls):
                if self.checkpoint > i:
                    t.update()
                    continue
                
                self.checkpoint=i
                desc= "articles {}/{} | Error {}".format(len(self.articles.list), len(self.urls), len(self.error_urls))
                t.update()
                t.set_description_str(desc)
                self.crawl(url)

    def crawl(self, url):
        self.browser.get(url)
        keywords = set()
        date=set()
        for instance in self.urls[url]:
            date.add(instance["date"])
            keywords.add(instance["keyword"])
        keywords = tuple(keywords)
        date = tuple(date)
        try:
            title = self.get_article_title()
            press = self.get_press_info()
            body = self.get_article_body()
            datetime = self.get_article_datetime()
            if len(self.articles.list) % 100 == 0: # savepoint (interval) = 100
                self.save_checkpoint()
            self.articles.add_article(title, keywords, date,
                                    datetime, press, url, body)
        except Exception:
            self.save_errors(url)
            
    def save_errors(self, url):
        self.error_urls.add(url)
        import pickle
        with open("./articles/error_urls_2.pkl","wb") as f:
            dmp = self.error_urls.copy()
            pickle.dump(dmp, f)
            dmp = None

    def get_article_datetime(self):
        try:
            datetime = self.browser.find_element_by_class_name("t11").text
        except Exception:
            datetime = self.browser.find_element_by_class_name("article_info")
            datetime = datetime.find_element_by_class_name("author").text
        return datetime

    def get_article_title(self):
        try:
            title = self.browser.find_element_by_id("articleTitle")
        except Exception:
            title = self.browser.find_element_by_class_name("end_tit")
        return title.text

    def get_press_info(self):
        press =  self.browser.find_element_by_class_name("press_logo")
        return press.find_element_by_tag_name("img").get_attribute("title")

    def get_article_body(self):
        try:
            body = self.browser.find_element_by_id("articleBodyContents")
        except Exception:
            body = self.browser.find_element_by_id("articeBody")
        return body.text
        
import pickle
urls = pickle.load(open("/home/ailee/SynologyDrive/SIRE/Projects/Crawler/Coronavirus Crawler/articles_url.pkl", "rb"))
print("{:,} source urls...".format(len(urls)))

import math
index_s = math.floor(len(urls) * (1/4)) # Second part
index_e = math.floor(len(urls) * (2/4)) 
url_keys = list(urls.keys())[index_s:index_e]

with open("./articles/parallel/urls_2.pkl", "wb") as w:
    dmp = url_keys.copy()
    pickle.dump(dmp, w)
    dmp=None

from tqdm import tqdm
tmp = {}
for key in tqdm(url_keys):
    tmp[key] = urls[key]

urls = tmp.copy()
tmp=None
print("{:,} urls - PARALLEL [2]".format(len(urls.keys())))
# For parallel crawling - END

crawler = Crawler(urls)
crawler.crawl_urls()
