import sys
import datetime
import sql
import psycopg2
import helpers


def process_file(filename, database):
    print('Processing file %s' % filename)
    cur = database.cursor()

    # tweet attributes
    tweet_timestamp = None
    tweet_user = None
    tweet_content = None

    # read the tweets
    tweet_number = 0
    first_line = True
    for line in helpers.lazy_read_file(filename):
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

            # add to a database
            tweets = sql.Table('tweets')
            query = tuple(tweets.insert(columns = [tweets.timestamp, tweets.user, tweets.content],
                                        values = [[tweet_timestamp, tweet_user, tweet_content]]))
            cur.execute(query[0], query[1])
            database.commit()

            tweet_number += 1
        elif first_line:
            # first line contains the tweet number, ignore
            first_line = False
        else:
            print('Unknown tweet attribute %s' % attr)
            exit(1)

    # close the cursor
    cur.close()

    # stats
    print('Inserted %d tweets into the database from file %s' % (tweet_number, filename))


# connect to postgres
db = None
try:
    db = psycopg2.connect(host = 'localhost')
except:
    print('Cannot connect to the database, is Postgres running?')
    exit(1)

# process files
for file in sys.argv[1:]:
    process_file(file, db)

# close database
db.close()

# stats
print('Processed %d files' % len(sys.argv[1:]))
