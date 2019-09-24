# -*- coding: utf-8 -*-

import wikipediaapi
import time
import random
import json

from selenium import webdriver


def fetch(q):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(q)
    categories = []
    for category in page_py.categories:
        categories.append(category)

    if not categories or not page_py:
        return None

    return {'name': q, 'summary': page_py.summary, 'categories': categories}


def generate_sample():
    famous_people = []
    with open('files/living_people.csv', 'r') as fs:
        for line in fs:
            famous_people.append(line.strip())

    training_set = random.sample(famous_people, 85000)

    # Test set should have no data from the training set
    for _ in training_set:
        famous_people.remove(_)

    test_set = random.sample(famous_people, 15000)

    with open('files/training_set.csv', 'a') as fs:
        for _ in training_set:
            fs.write(_ + '\n')

    with open('files/test_set.csv', 'a') as fs:
        for _ in test_set:
            fs.write(_ + '\n')


def scrape_names():
    with open('files/living_people.csv', 'w'):
        pass

    try:
        driver = webdriver.Chrome(r'./chromedriver')
        driver.get('https://en.wikipedia.org/wiki/Category:Living_people')

        for page in range(5000):
            print('Parsing page {}'.format(page))
            groups = driver.find_elements_by_xpath("//div[@class='mw-category-group']")

            with open('files/living_people.csv', 'a') as fs:
                for group in groups:
                    fs.write(group.text)
                    fs.write('\n')

            next_page = driver.find_element_by_xpath(
                '//*[@id="mw-pages"]/a[2]') if page == 0 else driver.find_element_by_xpath('//*[@id="mw-pages"]/a[4]')
            next_page.click()
            time.sleep(3)
    except Exception as e:
        print(e)
    finally:
        driver.quit()


def scrape_pages():
    # Training set
    names = []
    data = []
    js = {}
    with open('files/training_set.csv', 'r') as fs:
        for _ in fs:
            names.append(_)

    for name in names:
        try:
            data.append(fetch(name))
        except Exception:
            pass

    js['data'] = data
    with open('files/training_set.json', 'w') as fs:
        json.dump(js, fs)

    # Test set
    names = []
    data = []
    js = {}
    with open('files/test_set.csv', 'r') as fs:
        for _ in fs:
            names.append(_)

    for name in names:
        try:
            data.append(fetch(name))
        except Exception:
            pass

    js['data'] = data
    with open('files/test_set.json', 'w') as fs:
        json.dump(js, fs)


if __name__ == '__main__':
    # scrape_names()
    generate_sample()
