import wikipediaapi


def fetch(q):
    wiki_wiki = wikipediaapi.Wikipedia('en')

    page_py = wiki_wiki.page('Python_(programming_language)')
    print(page_py.summary)

    for category in page_py.categories:
        print(category)


def scrape():
    url = 'https://en.wikipedia.org/wiki/Category:Living_people'


if __name__ == '__main__':
    pass
