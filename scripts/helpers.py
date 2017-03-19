"""
Helpers for my scripts. Contains handy functions for working with tweets
and huge files, logging, etc.
"""

import io
import datetime


def lazy_read_file(filename):
    """
    A generator that reads a file line by line.
    """
    with io.open(filename, buffering = 1, encoding = 'utf-8') as file:
        for line in file:
            yield line


def log(string = '', log_file = 'log.txt'):
    """
    Log a string to the console and to a file.
    """
    time_string = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + str(string)

    print(time_string)
    with io.open(log_file, mode = 'a') as file:
        file.write(time_string + '\n')


def write_tweet(filename, tweet_timestamp, tweet_user, tweet_content):
    """
    Write a tweet to a file (append). Use the same format as the dataset.
    """
    with io.open(filename, mode = 'a') as file:
        file.write('T\t%s\n' % tweet_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        file.write('U\t%s\n' % tweet_user)
        file.write('W\t%s\n' % tweet_content)
        file.write('\n')


def read_tweet(file):
    """
    Read and parse a tweet from a file.
    :param file: The lazy file generator.
    """
    timestamp_line = ' '
    user_line = ' '
    content_line = ' '

    while timestamp_line[0] != 'T':
        timestamp_line = next(file)
    while user_line[0] != 'U':
        user_line = next(file)
    while content_line[0] != 'W':
        content_line = next(file)

    timestamp = datetime.datetime.strptime(timestamp_line[2:-1], '%Y-%m-%d %H:%M:%S')
    user = user_line[(user_line.rfind('/') + 1):-1]
    content = content_line[2:-1]

    return timestamp, user, content


def lazy_read_tweets(filename):
    """
    A lazy generator that reads tweets from a file.
    """
    file = lazy_read_file(filename)

    total_line = next(file)
    total = int(total_line[(total_line.rfind(':') + 1):-1])

    tweet_id = 0
    while True:
        tweet = read_tweet(file)
        yield tweet[0], tweet[1], tweet[2], tweet_id, total
        tweet_id += 1
