import pandas as pd
import requests
from bs4 import BeautifulSoup


dic = {
    'title':[],
    'url':[],
    'tags':[],
    'date':[],
    'preview':[],
    'body':[]
}


for idx in range(1, 10):

  # Main section URL for sports news
  base_url = '' # Articles are relative to this base URL
  # section_url = f"/{idx}" #sport
  # section_url = f"/{idx}" #accident
  # section_url = f"/{idx}" #world
  # section_url = f"/297/{idx}" #economy
  section_url = f"{idx}" #art

  response = requests.get(section_url)
  soup = BeautifulSoup(response.content, 'html.parser')

  articles = soup.select('div.bigOneSec')

  for article in articles:
      a_tag = article.select_one('a.bigOneImg')
      if a_tag:
          href = a_tag['href']
          article_url = base_url + href
          title = article.select_one('h3 a').get_text(strip=True)
          preview = article.select_one('p').get_text(strip=True)
          timestamp = article.get('data-id', 'N/A')

          dic['title'].append(title)
          dic['url'].append(article_url)
          dic['preview'].append(preview)
          dic['date'].append(timestamp)

          article_response = requests.get(article_url)
          article_soup = BeautifulSoup(article_response.content, 'html.parser')

          body_paragraphs = article_soup.select('div#articleBody p')
          article_body = '\n'.join([p.get_text(strip=True) for p in body_paragraphs])
          dic['body'].append(article_body)
          tags = []
          tags_section = article_soup.select_one('div.tags')
          tags = []
          if tags_section:
              tag_links = tags_section.select('h3 a')  
              for tag in tag_links:
                  tag_text = tag.get_text(strip=True)
                  if tag_text:
                      tags.append(tag_text)
          dic['tags'].append(tags)


df = pd.DataFrame(dic)
df.to_csv('art.csv', index=False)
