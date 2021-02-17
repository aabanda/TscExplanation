import requests
from bs4 import BeautifulSoup
import fasttext
import sys


#sys.setdefaultencoding('UTF8')
lid_model = fasttext.load_model("lid.176.ftz")








#La vanguardia:

URL = 'https://www.lavanguardia.com/vida/20200701/482046392639/juez-de-eeeuu-bloquea-regla-de-tercer-pais-para-quienes-piden-asilo.html'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')


Title= soup.title.text.strip()
Idioma = lid_model.predict(Title)[0][0].replace('__label__', '')

results =soup.find("div", class_="logo")
Periodico = results.get_text().strip()

results =soup.find("div", class_="author-name")
Autor = results.get_text().strip()

results =soup.find("div", class_="date-time")
Fecha = results.get_text().strip()[:10]

results =soup.find("div", class_="supra-title-container")
Antetitulo = results.get_text().strip()

results =soup.find("div", class_="supra-title-container")
Antetitulo = results.get_text().strip()

results =soup.find("div", class_="epigraph-container")
Subtitulo = results.get_text().strip()


URL

results =soup.find("div", class_="article-modules")
Contenido = results.get_text().strip()



#El Pais :

URL = 'https://elpais.com/internacional/elecciones-usa/2021-01-13/el-impeachment-revela-las-grietas-que-la-sublevacion-ha-abierto-en-el-partido-republicano.html'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')



results =soup.find("div", id="article_header")
count= 0
for el in results:
    if count==0:
        Antetitulo = el.get_text()
    elif count == 1:
        Titulo = el.get_text()
    elif count ==2:
        Subtitulo =  el.get_text()
    count =count+1

Idioma = lid_model.predict(Titulo)[0][0].replace('__label__', '')


#
# results =soup.find("div", class_="logo")
# Periodico = results.get_text().strip()

results =soup.find("a", class_="a_aut_n | color_black")
Autor = results.get_text().strip()


results =soup.find("a", class_="a_ti")
Fecha = results.get_text().strip()[:11]


URL

results =soup.find("div", class_="a_b article_body | color_gray_dark")
Contenido = results.get_text().strip()




#El mundo



URL = 'https://www.elmundo.es/madrid/2021/01/13/5fff4f7cfdddff7a7a8b4661.html'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')


#
# results =soup.find("div", class_="ue-c-article__kicker-group")
# count= 0
# for el in results:
#     if count==0:
#         Antetitulo = el.get_text()
#     elif count == 1:
#         Titulo = el.get_text()
#     elif count ==2:
#         Subtitulo =  el.get_text()
#     count =count+1
#
# Idioma = lid_model.predict(Titulo)[0][0].replace('__label__', '')
#
#
# #
# # results =soup.find("div", class_="logo")
# # Periodico = results.get_text().strip()
#
# results =soup.find("a", class_="a_aut_n | color_black")
# Autor = results.get_text().strip()
#
#
# results =soup.find("a", class_="a_ti")
# Fecha = results.get_text().strip()[:11]
#
#
# URL
#
# results =soup.find("div", class_="a_b article_body | color_gray_dark")
# Contenido = results.get_text().strip()
