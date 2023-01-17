import re
import numpy as np
import pandas as pd
from nltk import PorterStemmer
import os
import json


def load_data():
    # TODO 1: ucitati podatke iz data/train.tsv datoteke
    # rezultat treba da budu dve liste, texts i sentiments
    data = pd.read_csv('./../data/train.tsv', sep='\t')
    texts = data.text.values
    sentiments = data.sentiment.values
    return texts, sentiments


def load_newspaper_data():
    # path = './../data/20news-bydate-train/'  # Path za treniranje
    path = './../data/20news-bydate-test/'  # Path za testiranje
    texts = []
    sentiments = []
    dirs = os.listdir(path)
    sentiment_classes = dirs

    for dir in dirs:
        path1 = path + dir + '/'
        files = os.listdir(path1)
        for file in files:
            s = ''
            with open(path1+file, 'r') as f:
                for i in f:
                    s += i + ' '
            texts.append(s)
            sentiments.append(dir)
    return texts, sentiments, sentiment_classes


def preprocess(text):
    # TODO 2: implementirati preprocesiranje teksta
    # - izbacivanje znakova interpunkcije
    # - svodjenje celog teksta na mala slova
    # rezultat treba da bude preprocesiran tekst

    temp = text
    text = ''
    for i in temp:
        if re.match('[A-Za-z ]', i):
            text += i.lower()

    return text.strip()


def tokenize(text):
    ps = PorterStemmer()
    stop_words = pd.read_csv('./../data/stop_words.tsv').values

    text = preprocess(text)
    # TODO 3: implementirati tokenizaciju teksta na reci
    # rezultat treba da bude lista reci koje se nalaze u datom tekstu
    words = text.split()

    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return words


def count_words(text):
    words = tokenize(text)
    # TODO 4: implementirati prebrojavanje reci u datum tekstu
    # rezultat treba da bude mapa, ciji kljucevi su reci, a vrednosti broj ponavljanja te reci u datoj recenici
    words_count = {}
    for word in words:
        if word in words_count:
            words_count[word] += 1
        else:
            words_count[word] = 1
    return words_count


def find_sentiment_classes(sentiments):
    sentiment_classes = []
    for s in sentiments:
        if s not in sentiment_classes:
            sentiment_classes.append(s)

    return sentiment_classes


def fit(texts, sentiments, sentiment_classes):
    # inicijalizacija struktura
    bag_of_words = {}               # bag-of-words za sve recenzije
    words_count = {}
    texts_count = {}
    texts_per_sentiments = {}
    for sc in sentiment_classes:
        words_count[sc] = {}
        texts_count[sc] = 0.0
        texts_per_sentiments[sc] = ''

    # TODO 5: proci kroz sve recenzije i sentimente i napuniti gore inicijalizovane strukture
    # bag-of-words je mapa svih reci i broja njihovih ponavljanja u celom korpusu recenzija
    all_words = ''
    # all_pos = ''
    # all_neg = ''
    i = 0
    for text in zip(texts, sentiments):
        all_words += text[0] + ' '
        texts_per_sentiments[text[1]] += text[0] + ' '
        texts_count[text[1]] += 1
        i += 1
        if i%1000 == 0:
            print(i)
        """if text[1] == 'pos': Hardcode-ovane vrednosti kod klasifikacije recenzija filmova
            all_pos += text[0] + ' '
            texts_count['pos'] += 1
        else:
            all_neg += text[0] + ' '
            texts_count['neg'] += 1"""
    print(i)
    bag_of_words = count_words(all_words)
    print('all')
    for sc in sentiment_classes:
        words_count[sc] = count_words(texts_per_sentiments[sc])
        print(sc)
    # Hardcode-ovane vrednosti kod klasifikacije recenzija filmova
    # words_count['pos'] = count_words(all_pos)
    # words_count['neg'] = count_words(all_neg)
    return bag_of_words, words_count, texts_count


def predict(text, bag_of_words, words_count, texts_count, sentiment_classes):
    words = tokenize(text)          # tokenizacija teksta

    # TODO 6: implementirati Naivni Bayes klasifikator za sentiment teksta (recenzije)
    # rezultat treba da bude mapa verovatnoca da je dati tekst klasifikovan kao pozitivnu i negativna recenzija
    pi = {}
    number_of_all_words_i = {}
    scores = {}
    for sc in sentiment_classes:
        pi[sc] = texts_count[sc] / sum(texts_count.values())
        number_of_all_words_i[sc] = sum(words_count[sc].values())
        scores[sc] = 0

    # Hardcode-ovane vrednosti kod klasifikacije recenzija filmova
    # pp = texts_count['pos']/(texts_count['pos'] + texts_count['neg'])
    # pn = texts_count['neg']/(texts_count['pos'] + texts_count['neg'])
    # number_of_all_words_pos = sum(words_count['pos'].values())
    # number_of_all_words_neg = sum(words_count['neg'].values())

    number_of_all_words = sum(bag_of_words.values())

    for sc in sentiment_classes:
        suma = 0
        for word in words:
            pit = (words_count[sc].get(word, 0) + 1) / (number_of_all_words_i[sc] + number_of_all_words)
            pt = bag_of_words.get(word, 0) / number_of_all_words
            if pt > 0:
                suma += np.log(pit / pt)

        suma += np.log(pi[sc])
        scores[sc] = np.exp(suma)

    """suma = 0 Hardcode-ovane vrednosti kod klasifikacije recenzija filmova
    for word in words:
        ppt = (words_count['pos'].get(word, 0) + 1) / (number_of_all_words_pos + number_of_all_words)
        pt = bag_of_words[word] / number_of_all_words
        suma += np.log(ppt / pt)

    suma += np.log(pp)
    score_pos = np.exp(suma)
    suma = 0
    for word in words:
        ppn = (words_count['neg'].get(word, 0) + 1) / (number_of_all_words_neg + number_of_all_words)
        pt = bag_of_words[word] / number_of_all_words
        suma += np.log(ppn / pt)

    suma += np.log(pn)
    score_neg = np.exp(suma)"""
    max_value = max(scores.values())  # maximum value
    max_keys = [k for k, v in scores.items() if v == max_value]  # getting all keys containing the `maximum`

    return scores, (max_keys[0], max_value)


if __name__ == '__main__':
    # ucitavanje data seta
    texts, sentiments, sentiment_classes = load_newspaper_data()
    # sentiment_classes = find_sentiment_classes(sentiments)

    # izracunavanje / prebrojavanje stvari potrebnih za primenu Naivnog Bayesa
    # bag_of_words, words_count, texts_count = fit(texts, sentiments, sentiment_classes)
    """with open('trained.json', 'w') as f: Kreiranje fajla sa vrednostima posle treniranja
        json.dump((bag_of_words, words_count, texts_count), f)"""
    with open('trained.json', 'r') as f:  # Preuzimanje vrednosti prethodno istreniranih parametara
        bag_of_words, words_count, texts_count = json.load(f)
    # recenzija
    # text = 'I dont know who made this movie, but its awesome!'

    # klasifikovati sentiment recenzije koriscenjem Naivnog Bayes klasifikatora
    success = 0
    all_tries = 0
    for i in zip(texts, sentiments):
        all_tries += 1
        predictions = predict(i[0], bag_of_words, words_count, texts_count, sentiment_classes)
        if i[1] == predictions[1][0]:
            success += 1
        if all_tries % 1000 == 0:
            print('Accuracy: {:0.2f}%'.format(success*100/all_tries))

    print('Final accuracy: {:0.2f}%'.format(success*100/all_tries))  # 75.46468401486989 tacnost

    # predictions = predict(text, bag_of_words, words_count, texts_count, sentiment_classes)

    """print('-'*30) Ispis za klasifikaciju recenzija filmova
    print('Review: {0}'.format(text))
    print('Score(pos): {0}'.format(predictions['pos']))
    print('Score(neg): {0}'.format(predictions['neg']))"""
