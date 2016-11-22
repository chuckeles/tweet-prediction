import io
import datetime


def lazy_read_file(filename):
    with io.open(filename, buffering = 1, encoding = 'utf-8') as file:
        while file.readable():
            yield file.readline()


def log(string, log_file = 'log.txt'):
    time_string = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + string

    print(time_string)
    with io.open(log_file, mode = 'a') as file:
        file.write(time_string + '\n')


def add_tweet(filename, tweet_timestamp, tweet_user, tweet_content):
    with io.open(filename, mode = 'a') as file:
        file.write('T\t%s\n' % tweet_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        file.write('U\t%s\n' % tweet_user)
        file.write('W\t%s\n' % tweet_content)
        file.write('\n')
