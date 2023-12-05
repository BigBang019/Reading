import urllib.parse
import urllib.request
import re
import json
from collections import defaultdict

def ParseCookie(s):
    cookie = {}
    for ss in s.split("; "):
        res = ss.split('=')
        if len(res) > 1:
            cookie[res[0]] = '='.join(res[1:])
    return cookie

class Cookie():
    def __init__(self, cookie):
        self.cookie = {}
        if isinstance(cookie, dict):
            self.cookie = cookie
        if isinstance(cookie, str):
            self.cookie = ParseCookie(cookie)

    def update(self, s):
        res = s.split('; ')[0].split('=')
        self.cookie[res[0]] = ''.join(res[1:])

    def clear(self):
        self.cookie = {}
    
    def __str__(self):
        return '; '.join([f'{k}={v}' for k, v in self.cookie.items()])

def HtmlReplace(html):
    html = re.sub(r'&amp;', "&", html)
    html = re.sub(r'<b>', "", html)
    html = re.sub(r'</b>', "", html)
    return html

class Scholar:
    def __init__(self):
        cs = 'SEARCH_SAMESITE=CgQI85cB; AEC=AUEFqZfCHurDwCAFt9Jhad0HBRW1Pq7TROlw_hpUGN0IfaMnSEWEHajP_w; 1P_JAR=2023-3-27-15; SID=Uwg6or5v50v6RKtb6Doz48zEPHD_NGfgw1ppRMlCxCLYgdJYgDWRUNSZgkeah-RRPSmASw.; __Secure-1PSID=Uwg6or5v50v6RKtb6Doz48zEPHD_NGfgw1ppRMlCxCLYgdJYewFBvZaXnz4B62pU0ETXkA.; __Secure-3PSID=Uwg6or5v50v6RKtb6Doz48zEPHD_NGfgw1ppRMlCxCLYgdJYsnNHT3zhb7UVw4YwckQrBw.; HSID=AUH5TOdQph6Fy05E-; SSID=A0s6CCpqB0in37GhI; APISID=zHK_R9QiHxZ0CS-N/A1VyZ17pArG0kAE4d; SAPISID=IKE-SEywCk1V8I4h/AiwGr-y10PZAapNe-; __Secure-1PAPISID=IKE-SEywCk1V8I4h/AiwGr-y10PZAapNe-; __Secure-3PAPISID=IKE-SEywCk1V8I4h/AiwGr-y10PZAapNe-; GSP=LM=1679929902:S=gHDD-b7lEjuekq5I; __Secure-ENID=11.SE=SUNq92jR1bft5jCWqCvZ0FhSP1kPTfabY5nCQU6WSi5HWFAQNd4E4mQWWpEXATaUonoXoB-IfbgwOu9q0FDczRibO3ASNsRCUy3GJFa_cClHJ8SJMJ5cmhelaaaXO9WBLnccGqKIdJv0u0Y1QmaPXWnK70rIkJCVymSo7gvmkjaw_7lulRa73cfoqQamcvqP0oCpXPPEW4dd9BCd5cMF6ZXJB0japAv_eDe5LHOVjX8TsoIWNCUnwzqJiX1aaOSiC-H0a5XOSEDaHzr6zfUPYos7_aMthw; SIDCC=AFvIBn-g8ML0rUmSlLEOTexLbzDriSgKczlIfy3dNhRhGk57SwPPVb-kXPqyJ_727tBpZbXv; __Secure-1PSIDCC=AFvIBn9xugHa1kO7v5y30JOQWTawwQulZfRV5pC3BqzhwmpgcjWwo6JgIyB-QVISzIKA9r9N; __Secure-3PSIDCC=AFvIBn9UpJ9NV_6v-CuT5z1H4o4qakWTQT-xu_nkZRrKRWP2h9bkNLT6vYg9EFH-a_3vrzvQ; NID=511=JjHxwTIpKEi8vecD-oVxd0zDySWA_ET8QjCOyq54blUTkHtTMTuLwHVakuJ24NVcCN-YzdKDo6R1wiD4Qu3uscKNkVWVYtBzMBg2AyzHHiFKDjy5gxoWQn48-7f_1AqyNU18vTLjHrSDV4z2tw1gK-2_2m5sD4BEdjnt6DhpDEjU0rCbgeCvHMNJ2M3VAmNIppS4nVyi2axqjoGbwmfb7fhI0fS1IUkgu0njEJILfGWHUcEGnbHB3aDUuCYjX3Ru6TzwjpA'
        self.cookie = Cookie("")
    
    def _GetData(self, url):
        header_dict={
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            # 'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,fr;q=0.6,zh-TW;q=0.5,ja;q=0.4',
            'cache-control': 'max-age=0',
            'cookie': str(self.cookie),
            'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
            'x-client-data': 'CIa2yQEIpbbJAQjBtskBCKmdygEI3tPKAQil8MoBCJShywEImv7MAQjI/8wBCLGPzQEIhJPNAQiQls0BCOGXzQEI5JfNAQjomc0BCOmbzQEIy5zNAQiwn80BCL+fzQE=',
        }
        req = urllib.request.Request(url=url,headers=header_dict)
        response = urllib.request.urlopen(req, timeout=120)

        tempdata = response.read()
        for k, v in response.headers.items():
            if k == 'Set-Cookie':
                self.cookie.update(v)
        tempdata = tempdata.decode()
        return tempdata

    def GetData(self, url):
        return self._GetData(url)

    def ParseTitleFromHtml(self, html):
        title_pattern = re.compile(r'<a id="[^"]*" href="[^"]*" data-clk="[^"]*" data-clk-atid="[^"]*">.*</a>')
        titles = re.finditer(title_pattern, html)
        for title in titles:
            print(title)
            title_str = title.group(1)
            print(title_str)
            title_str = HtmlReplace(title_str)
            print(title_str)
        return titles

class DBLP:
    def __init__(self):
        pass

    def GetData(self, url):
        req = urllib.request.Request(url=url)
        response = urllib.request.urlopen(req, timeout=120)
        tempdata = response.read()
        tempdata = tempdata.decode()
        return tempdata
    
    def GetPubFromTitle(self, title):
        url = "https://dblp.org/search/publ/api?q=" + urllib.parse.quote(title) + "&format=json"
        pub = self.GetData(url)
        pub = json.loads(pub)
        return pub['result']['hits']['hit'][0]['info']['venue']
    
    def GetAuthorFromVenue(self, venue, filter_affilation=None):
        url = "https://dblp.org/search/publ/api?q=" + urllib.parse.quote(f'venue:{venue}') + "&format=json&h=1000"
        pub = self.GetData(url)
        pub = json.loads(pub)
        hits = pub['result']['hits']['hit']
        found_author = []
        for hit in hits:
            if 'authors' in hit['info'].keys():
                authors = hit['info']['authors']['author']
                if not isinstance(authors, list):
                    continue
                for author in authors:
                    name = author['text']
                    affilation = self.GetAffilationFromAuthor(name)
                    if len(affilation)==0:
                        continue
                    affilation = affilation[0]
                    if filter_affilation is not None and filter_affilation in affilation:
                        found_author.append((name, affilation))
        return found_author

    def GetAffilationFromAuthor(self, author):
        url = "https://dblp.org/search/author/api?q=" + urllib.parse.quote(author) + "&format=json&h=1000"
        author = self.GetData(url)
        author = json.loads(author)
        hits = author['result']['hits']['hit']
        found_affilations = []
        for hit in hits:
            if 'notes' in hit['info'].keys():
                note = hit['info']['notes']['note']
                if isinstance(note, list):
                    note = note[0]
                if note['@type'] == 'affiliation':
                    found_affilations.append(note['text'])
        return found_affilations

if __name__ == "__main__":
    dblp = DBLP()
    scholar = Scholar()
    pubs_cnt = defaultdict(lambda:0)
    
    '''dblp'''
    author_affilations = dblp.GetAuthorFromVenue("AAAI", "Hong Kong")
    print('hi')
    exit()
    with open('pub_cnt.json', 'w') as f:
        json.dump(pubs_cnt, f)