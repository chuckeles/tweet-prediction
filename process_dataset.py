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
import nltk
import sql
import multiprocessing as mp


processed_count = None
inserted_count = None
errored_count = None


def process_file(filename):
    """
    Process one dataset file concurrently.
    """
    with mp.Pool() as pool:
        pool.map(process_tweet, helpers.lazy_read_tweets(filename))


def process_tweet(tweet):
    """
    Process one tweet. This function is run by a worker.
    """
    global processed_count, inserted_count, errored_count

    try:
        with processed_count.get_lock():
            processed_count.value += 1

        # decompose
        timestamp, user, content, tweet_id, total = tweet

        # print progress
        if tweet_id % 2000 == 0:
            helpers.log('Processed %.3f%% of tweets' % (tweet_id / total))

        # guess the tweet language
        language = gl.guess_language(content)
        if language != 'en':
            return

        # tokenize
        tokens = nltk.tokenize.casual_tokenize(content, preserve_case = False, reduce_len = True)

        # filter out some punctuation and stop words
        punc = ['.', ',', '!', '?']
        stop_words = nltk.corpus.stopwords.words('english')
        tokens = map(lambda token: ''.join(filter(lambda char: char not in punc, token)), tokens)
        tokens = filter(lambda token: token and token not in stop_words, tokens)
        tokens = list(tokens)

        # stemming
        stemmer = nltk.stem.PorterStemmer()
        stems = map(lambda token: stemmer.stem(token), tokens)
        stems = list(set(stems))

        # store in the database
        db = pg.connect(host = 'localhost')
        cur = db.cursor()

        table = sql.Table('tweets')
        query = tuple(table.insert(columns = [table.timestamp, table.user, table.content, table.tokens, table.stems],
                                   values = [[timestamp, user, content, tokens, stems]]))

        cur.execute(query[0], query[1])
        db.commit()
        db.close()

        with inserted_count.get_lock():
            inserted_count.value += 1

    except Exception as e:
        helpers.log('Error processing tweet:')
        helpers.log(e)

        with errored_count.get_lock():
            errored_count.value += 1


if __name__ == '__main__':
    """
    Process the files on the input and show final statistics.
    """
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
