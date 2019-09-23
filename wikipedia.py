import wikipediaapi
import time

from selenium import webdriver


def fetch(q):
    wiki_wiki = wikipediaapi.Wikipedia('en')

    page_py = wiki_wiki.page('Python_(programming_language)')
    print(page_py.summary)

    for category in page_py.categories:
        print(category)


def scrape():
    with open('living_people.csv', 'w'):
        pass

    try:
        driver = webdriver.Chrome(r'./chromedriver')
        driver.get('https://en.wikipedia.org/wiki/Category:Living_people')

        for page in range(5000):
            print('Parsing page {}'.format(page))
            groups = driver.find_elements_by_xpath("//div[@class='mw-category-group']")

            with open('living_people.csv', 'a') as fs:
                for group in groups:
                    fs.write(group.text)
                    fs.write('\n')

            next_page = driver.find_element_by_xpath(
                '//*[@id="mw-pages"]/a[2]') if page == 0 else driver.find_element_by_xpath('//*[@id="mw-pages"]/a[4]')
            next_page.click()
            time.sleep(1)
    except Exception as e:
        print(e)
    finally:
        driver.quit()


if __name__ == '__main__':
    scrape()
