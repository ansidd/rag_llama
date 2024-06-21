from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import xml.etree.ElementTree as ET


#Extract plain text from html content
def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.body.get_text(separator=" ")

    lines = (line.strip() for line in text.splitlines())

    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    text = '\n'.join(chunk for chunk in chunks if chunk)  
    return text

#parses sitemap to extract urls that can be scrapped
def parseXML(xmlfile): 
    tree = ET.parse(xmlfile)
    urls=[]
    for child in tree.getroot():
        urls.append(child[0].text)
    return urls

class AppURLopener(urllib.request.FancyURLopener):
    version = "App/1.7"


#extract text from each url and store it in text files
if __name__=="__main__":

    urls = parseXML("./sitemap.xml")
    with open("urls.txt", "w") as f:
        f.write("\n".join(urls))

    for i,url in enumerate(urls):

        try:
            req = urllib.request.Request(
                url=url,
                headers={'User-Agent': 'Mozilla/6.0'}
            )
            
            html = urllib.request.urlopen(req).read()
            text = text_from_html(html)
            with open("./data/file_"+str(i)+".txt", "w") as f:
                f.write(text)
            
        except Exception as e:
            print(url, e)




