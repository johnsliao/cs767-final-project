import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

page_py = wiki_wiki.page('Python_(programming_language)')
print(page_py.summary)
print(page_py.categories)
