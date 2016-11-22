import sys
import datetime
import sql
import psycopg2
import nltk
from nltk.corpus import stopwords
import helpers


def process_file(filename, database):
    helpers.log('Processing file %s' % filename)
    cur = database.cursor()

    # tweet attributes
    tweet_timestamp = None
    tweet_user = None
    tweet_content = None

    # stats
    first_line = True
    total_amount = 0
    tweet_count = 0
    english_tweets = 0
    inserted_tweets = 0

    # read the tweets
    for line in helpers.lazy_read_file(filename):
        # first line contains the number of tweets
        if first_line:
            first_line = False
            pos = line.rfind(':')
            total_amount = int(line[(pos + 1):-1])
            continue

        # is this an attribute?
        if len(line) < 3:
            continue

        # get the first char
        attr = line[0]

        # add the attribute
        if attr == 'T':
            tweet_timestamp = datetime.datetime.strptime(line[2:-1], '%Y-%m-%d %H:%M:%S')
        elif attr == 'U':
            last_slash = line.rfind('/')
            tweet_user = line[(last_slash + 1):-1]
        elif attr == 'W':
            tweet_content = line[2:-1]

            # print progress
            if tweet_count % 10000 == 0:
                helpers.log('Processed %.2f%% of tweets' % (tweet_count / total_amount * 100))

            # check the language
            tokens = nltk.wordpunct_tokenize(tweet_content)
            words = set(word.lower() for word in tokens)

            ratios = {}
            for language in stopwords.fileids():
                stopwords_set = set(stopwords.words(language))
                common_words = words.intersection(stopwords_set)
                ratios[language] = len(common_words)

            tweet_language = max(ratios, key = ratios.get)
            if tweet_language == 'english':
                english_tweets += 1

                # add to a database
                tweets = sql.Table('tweets')
                query = tuple(tweets.insert(columns = [tweets.timestamp, tweets.user, tweets.content],
                                            values = [[tweet_timestamp, tweet_user, tweet_content]]))
                try:
                    cur.execute(query[0], query[1])
                    database.commit()
                    inserted_tweets += 1
                except psycopg2.OperationalError as e:
                    helpers.log('Failed to insert the tweet number %d:' % tweet_count)
                    helpers.log(e)

                    helpers.add_tweet('failed_tweets.txt', tweet_timestamp, tweet_user, tweet_content)

            tweet_count += 1
        else:
            helpers.log('Unknown tweet attribute %s' % attr)

    # close the cursor
    cur.close()

    # print stats
    helpers.log()
    helpers.log('Stats:')
    helpers.log('Dataset: %s' % filename)
    helpers.log('Total tweets: %d' % tweet_count)
    helpers.log('English tweets: %d' % english_tweets)
    helpers.log('Inserted tweets: %d' % inserted_tweets)


# connect to postgres
db = None
try:
    db = psycopg2.connect(host = 'localhost')
except psycopg2.OperationalError as e:
    helpers.log('Failed to connect to the database:')
    helpers.log(e)
    exit(1)

# process files
for file in sys.argv[1:]:
    process_file(file, db)

# close database
db.close()

# stats
helpers.log('Processed %d files' % len(sys.argv[1:]))

