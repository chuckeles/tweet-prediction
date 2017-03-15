# Bachelor's thesis project

My project for the bachelor's thesis is predicting the user activity on Twitter using tweets and machine learning. This repository contains the source code.

# Installation

Install Python and [Postgresql](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-16-04). Don't forget to install `python-dev python3-dev libpq-dev`.

Then, install `pip3` packages: `pip3 install setuptools nltk numpy pandas scikit-learn psycopg2 guess_language-spirit python-sql --user`.

I'm using [Jupyter](http://jupyter.org/) for working and playing with the data so install that: `pip3 install jupyter`

# Database

Create a new table for the processed tweets:

```sql
CREATE TABLE "tweets" (
    "id" serial,
    "timestamp" timestamp NOT NULL,
    "user" text NOT NULL,
    "length" integer NOT NULL,
    "words" integer NOT NULL,
    "hashtags" text[] DEFAULT '{}',
    "mentions" text[] DEFAULT '{}',
    "urls" text[] DEFAULT '{}',
    PRIMARY KEY ("id")
);
```

I'm using a transformed view of the data, create it like this:

```sql
CREATE VIEW "grouped_tweets" AS SELECT
    date_trunc('week', tweets."timestamp") AS week,
    tweets."user",
    count(tweets."id") AS tweets,
    sum(tweets."length") AS total_length,
    sum(tweets."words") AS total_words,
    array_cat_agg(tweets."hashtags") AS hashtags,
    array_cat_agg(tweets."mentions") AS mentions,
    array_cat_agg(tweets."urls") AS urls
  FROM tweets
  GROUP BY tweets."user", date_trunc('week', tweets."timestamp");
```

It's using this custom aggregate:

```sql
CREATE AGGREGATE array_cat_agg(anyarray) (
  SFUNC=array_cat,
  STYPE=anyarray
);
```

# Password

Use a `.pgpass` file to store the password for the database. Don't forget to `chmod 0600 .pgpass`. [More information in the docs](https://www.postgresql.org/docs/current/static/libpq-pgpass.html).
