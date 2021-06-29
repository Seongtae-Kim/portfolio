class Article:
    title=""
    body=""
    date=""
    url=""

def daterange(ds, de):
    from datetime import timedelta
    for n in range(int((de - ds).days)+1):
        yield ds + timedelta(n)

def dates_creator(ds, de):
    from datetime import timedelta, date
    dates = []
    date_start = date(ds[0], ds[1], ds[2])
    date_end = date(de[0], de[1], de[2])
    for date in daterange(date_start, date_end):
        dates.append(date.strftime("%Y.%m.%d"))
    print("dates:", len(dates))
    return dates

class Crawler:
    
    def __init__(self, ds, de, load=False):        
        self.checkpoint = self.create_checkpoint(load)
        self.sourceurls = self.create_sourceurls(ds, de)
        self.estimate_source_urls()
    
    def crawl(self, save_interval):
        print("Crawling Start...")
        self.browse_naver_news(savepoint=save_interval)
    
    def save_articlesurl(self, index1, index2, index3):
        import pickle
        with open("./checkpoint_index", "w") as writer:
            writer.write(str(index1)+" "+str(index2)+" "+str(index3))
        dmp = self.articleurls.copy()
        with open("./articles_url.pkl", "wb") as f:
            pickle.dump(dmp, f)
        print("\tSaving checkpoint...")
    
    def load_articlesurl(self):
        import pickle
        print("Loading last checkpoint...")
        with open("./articles_url.pkl", "rb") as f:
            self.articleurls = pickle.load(f)
            print("{:,} urls loaded".format(len(self.articleurls)))
        index1, index2, index3 = open("./checkpoint_index", "r").read().split()
        return int(index1), int(index2), int(index3)
        
    def browse_naver_news(self, savepoint=100, checkpoint_index=0):
        self.articleurls={}
        from selenium.webdriver.firefox.options import Options
        import selenium
        from selenium import webdriver
        from tqdm import tqdm
        import os, re
        if "checkpoint_index" in os.listdir("./"):
            index1, index2, index3  = self.load_articlesurl()
            i, j, k = index1, index2, index3
        else:
            index1, index2, index3 = -1, -1, -1
            
        options = Options()
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox") # ONLY ON UBUNTU!
        options.add_argument("--headless")
        browser = webdriver.Firefox(options=options)
        browser.minimize_window()
        
        # Keyword loop
        for i, keyword in enumerate(self.sourceurls):
            if index1 > i:
                continue
            print("keyword ", keyword, " ", i+1, "/", len(self.sourceurls))
            # Date loop
            for j, date in enumerate(self.sourceurls[keyword]):
                
                ##################################################
                if int(re.sub("\.", "", date)) < 20201229:
                    continue
                ##################################################

                if index1 <= i and index2 > j:
                    continue
                print("\tdate ", str(date), " | ", j+1, "/", len(self.sourceurls[keyword]), " | num articles :", len(self.articleurls))
                
                if j % 30 == 0 and j != 0:
                    self.save_articlesurl(i, j, k)
                self.lastpage = False
                
                # Page loop
                for k, page in enumerate(tqdm(self.sourceurls[keyword][date])):
                    if index1 <= i and index2 <= j and index3 > k:
                        continue
                    
                    # Savepoint
                    if (page % savepoint == 0 and page != 0):
                        self.save_articlesurl(i, j, k)
                    
                    if (page % 100 == 0 and page != 0):
                        browser.close()
                        browser = webdriver.Firefox(options=options)
                        browser.minimize_window()
                    
                    url = self.sourceurls[keyword][date][page]
                    
                    try:
                        if not self.is_obsolete_page(browser):
                            while True:
                                try:
                                    browser.get(url)
                                    break
                                except selenium.common.exceptions.NoSuchElementException:
                                    browser.close()
                                    browser = webdriver.Firefox(options=options)
                                    browser.minimize_window()
                                    pass
                            if self.has_search_result(browser):
                                self.get_articles_urls(browser, keyword)
                            else:
                                break
                        else:
                            break
                    except Exception:
                        while True:
                            try:
                                browser.get(url)
                                break
                            except selenium.common.exceptions.NoSuchElementException:
                                browser.close()
                                browser = webdriver.Firefox(options=options)
                                browser.minimize_window()
                                pass
                        if self.has_search_result(browser):
                            self.get_articles_urls(browser, keyword)
                        else:
                            break
                # end of k loop
                k = 0
                index3 = 0
            
            # end of j loop
            j = 0
            index2 = 0            
            self.save_articlesurl(i, j, k)
    
        self.save_articlesurl(i, j, k) # Last save
        print("Crawling complete")

    def is_obsolete_page(self, browser):
        buttons_wrap = browser.find_element_by_class_name("sc_page_inner")
        buttons = buttons_wrap.find_elements_by_class_name("btn")
        last =""
        for button in buttons:
            last = button.get_attribute("aria-pressed")
        if last == "true" and self.lastpage:
            return True # This page is obsolete
        elif last == "true" and not self.lastpage:
            self.lastpage = True # The next page will be obsolete
            return False # This page needs to be crawled 
        else:
            return False # This page needs to be crawled 

    def has_search_result(self, browser):
        try:
            list_news = browser.find_element_by_class_name("list_news")
            return True
        except Exception:
            print("\tNo Search Results")
            return False
        
    def get_articles_urls(self, browser, keyword): # per one source url
        import re
        temp_urls={}
        list_news = browser.find_element_by_class_name("list_news")
        
        #####
        info_groups = list_news.find_elements_by_class_name("info_group")
        for info_group in info_groups:
            a_s = info_group.find_elements_by_tag_name("a")
            url = ""
            date = ""
            for a in a_s:
                url_candid = a.get_attribute("href")
                if re.search("https://news.naver.com/", url_candid):
                    url = url_candid

            spans = info_group.find_elements_by_tag_name("span")
            for span in spans:
                date_candid = span.text
                if re.search("[0-9]+\.[0-9]+\.[0-9]+\.", date_candid):
                    date_candid = re.sub("\.","", date_candid)
                    date = date_candid

            if url != "" and date != "":
                if url not in self.articleurls.keys():
                    temp_urls[url] = [{"date" : date, "keyword" : keyword}]
                elif keyword not in [instance["keyword"] for instance in self.articleurls[url]]:
                    self.articleurls[url].append({"date" : date, "keyword" : keyword})
        ####
        sub_areas = list_news.find_elements_by_class_name("sub_area")
        for sub_area in sub_areas:
            a_s = info_group.find_elements_by_tag_name("a")
            url = ""
            date = ""
            for a in a_s:
                url_candid = a.get_attribute("href")
                if re.search("https://news.naver.com/", url_candid):
                    url = url_candid

            spans = info_group.find_elements_by_tag_name("span")
            for span in spans:
                date_candid = span.text
                if re.search("[0-9]+\.[0-9]+\.[0-9]+\.", date_candid):
                    date_candid = re.sub("\.","", date_candid)
                    date = date_candid

            if url != "" and date != "":
                if url not in self.articleurls.keys():
                    temp_urls[url] = [{"date" : date, "keyword" : keyword}]
                elif keyword not in [instance["keyword"] for instance in self.articleurls[url]]:
                    self.articleurls[url].append({"date" : date, "keyword" : keyword})
                    
        self.articleurls.update(temp_urls)
    
    
    def estimate_source_urls(self):
        cnt=0
        loop = self.sourceurls
        self.sourcequeue=[]
        for keywords in loop:
            for date in loop[keywords]:
                for page in loop[keywords][date]:
                    cnt+=1
                    self.sourcequeue.append(loop[keywords][date][page])
        print("source urls:", cnt)
        
    # maximum 4000 news articles per one keyword 
    def create_sourceurls(self, ds, de, page=3990): # page == min: 1 / max: 3990 / interval: 10
        sourceurls={}
        duration = dates_creator(ds, de)
        
        for query in self.checkpoint.keys():
            if query not in sourceurls:
                sourceurls[query] = {}

            for date in duration:
                if date not in sourceurls[query]:
                        sourceurls[query][date]={}
                        
                for p in range(0, page, 10):
                    if p not in sourceurls[query][date]:
                        sourceurls[query][date][p]=""
                        
                    url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds="+ date +"&de=" + date +"&start=" + str(p)
                    sourceurls[query][date][p] = url
        return sourceurls
        
    def create_checkpoint(self, load):
        import pickle
        
        keywords = ["사랑제일교회", "사회적 거리두기", "드라이브 스루", "대구", "우한", "후베이성",
                "정은경", "거브러여수스", "빌 게이츠", "이만희", "코로나19",
                "판데믹", "감염학계", "WHO", "CDC", "의협", "신종 코로나","덕분에 챌린지", 
                "K방역", "마스크", "신천지", "집합금지", "예배", "언택트", "입국제한",
                "비대면", "공중 보건", "여행 제한",  "전광훈", "바이러스", "감염병", "감염자",
                "확진자", "손소독제"]
        if load:
            checkpoint = pickle.load(open("./checkpoint.pkl", "rb"))    
        
        else:    
            checkpoint={}

            for keyword in keywords:
                checkpoint[keyword] = {"Completed":{"ds":"0000.00.00", "de":"0000.00.00"},
                                       "current":{"ds":"0000.00.00", "de":"0000.00.00"},
                                       "goal":{"ds":"2020.01.01", "de":"2020.12.31"}
                                      }
            with open("./checkpoint.pkl", "wb") as writer:
                pickle.dump(checkpoint, writer)
            checkpoint = pickle.load(open("./checkpoint.pkl", "rb"))    
        return checkpoint

crawler = Crawler((2020, 1, 1), (2020, 12, 31))
crawler.crawl(save_interval=1990)