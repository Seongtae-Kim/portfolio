alphabets=[]
a = 'a'
for i in range(26):
    alphabets.append(a)
    a = chr(ord(a) + 1)
print(alphabets)

def new_browser():
    from selenium.webdriver import Firefox, FirefoxOptions
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    options = Options()
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox") # ONLY ON UBUNTU!
    options.add_argument("--headless")
    return webdriver.Firefox(options=options)

def process(text, dic):
    text = text.split("\n")
    for i, word in enumerate(text):
        if i != len(text)-1:
            if i % 2 == 0:
                term_en = text[i]
                term_ko = text[i+1]
                dic[term_en] = term_ko
    return dic

from selenium.webdriver import Firefox, FirefoxOptions
from seongtae_utils import object_saveloader
from tqdm import tqdm
import pickle

browser = new_browser()
dic = {}
cnt=0
path = "/home/seongtae/SynologyDrive/SIRE/Projects/KUMC/terms"
path = "/home/ailee/SynologyDrive/SIRE/Projects/KUMC/terms"
dic = pickle.load(open(path+"/terms.pkl", "rb"))##
obj_saver = object_saveloader("terms", savepath=path)
with tqdm(alphabets, leave=False, bar_format="{percentage:2.2f}% {bar} {desc} {remaining}") as t:
    for alphabet in alphabets:
        if ord(alphabet) < ord("e"):
            t.update
            continue
        else:
            t.update()
        pageno=1
        endOfPage=False
        while not endOfPage:
            if alphabet == "e" and pageno < 1694:
                pageno=1694
            
            t.set_description_str("{} |  pageNO:{} | current terms: {} |".format(alphabet, pageno, len(dic)))
            url = "http://term.kma.org/search/list.asp?pageno={}&sch_txt={}&sch_type1=twdContent7&sch_type2=twdDong7".format(pageno, alphabet)
            cnt+=1
            if cnt % 100 ==0:
                browser.close()
                browser = new_browser()
            browser.get(url)
            text = browser.find_element_by_class_name("searchList").text
            if text == "":
                endOfPage=True
            else:
                dic = process(text, dic)
                obj_saver.index_update(dic, len(dic))
                pageno+=1