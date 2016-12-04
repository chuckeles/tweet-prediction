"""
The new script for processing the dataset. Loads the tweets,
processes the contents using natural language processing, and stores
them in the database. Concurrent, uses all CPU cores.
"""

import helpers
import psycopg2 as pg
import sys
import time
import guess_language as gl
import sql
import multiprocessing as mp
import itertools as it
import re


processed_count = None
inserted_count = None
errored_count = None

db = None


def process_file(filename):
    """
    Process one dataset file concurrently.
    """
    with mp.Pool() as pool:
        tweets = helpers.lazy_read_tweets(filename)

        # process tweets in chunks
        while True:
            res = pool.map(process_tweet, it.islice(tweets, mp.cpu_count()))
            if not res:
                break


def process_tweet(tweet):
    """
    Process one tweet. This function is run by a worker.
    """
    global processed_count, inserted_count, errored_count, db

    try:
        with processed_count.get_lock():
            processed_count.value += 1

        # decompose
        timestamp, user, content, tweet_id, total = tweet

        # print progress
        if tweet_id % 2000 == 0:
            helpers.log('Processed %.3f%% of tweets' % (tweet_id / total))

        # remove empty tweets
        if content == 'No Post Title':
            return

        # guess the tweet language
        language = gl.guess_language(content)
        if language != 'en':
            return

        # make stats
        url_regex = re.compile('^https?://')

        length = len(content)
        words = list(filter(lambda word: word, content.split(' ')))
        word_count = len(words)
        hashtags = list(map(lambda word: word[1:], filter(lambda word: word[0] == '#', words)))
        mentions = list(map(lambda word: word[1:], filter(lambda word: word[0] == '@', words)))
        urls = list(filter(lambda word: url_regex.match(word), words))

        # store in the database
        if not db:
            db = pg.connect(host = 'localhost')

        cur = db.cursor()

        table = sql.Table('tweets')
        query = tuple(
            table.insert(
                columns = [table.timestamp, table.user, table.length, table.words, table.hashtags, table.mentions,
                           table.urls],
                values = [[timestamp, user, length, word_count, hashtags, mentions, urls]]))

        cur.execute(query[0], query[1])
        db.commit()

        with inserted_count.get_lock():
            inserted_count.value += 1

    except Exception as e:
        helpers.log('Error processing tweet:')
        helpers.log(e)

        with errored_count.get_lock():
            errored_count.value += 1


def main():
    """
    Process the files on the input and show final statistics.
    """
    global processed_count, inserted_count, errored_count

    # stats
    processed_count = mp.Value('i', 0)
    inserted_count = mp.Value('i', 0)
    errored_count = mp.Value('i', 0)

    # process all input files
    for file in sys.argv[1:]:
        process_file(file)

    # print stats
    m, s = divmod(time.process_time(), 60)
    h, m = divmod(m, 60)
    helpers.log('Processed files: %d' % len(sys.argv[1:]))
    helpers.log('Processed tweets: %d' % processed_count.value)
    helpers.log('Errored tweets: %d' % errored_count.value)
    helpers.log('Inserted tweets: %d' % inserted_count.value)
    helpers.log('Total run time: %d:%d:%.3f' % (h, m, s))


if __name__ == '__main__':
    main()
