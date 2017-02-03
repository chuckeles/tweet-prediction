# Bachelor's thesis project

My project for the bachelor's thesis is predicting the user activity on Twitter using tweets and machine learning. This repository contains the source code.

# Installation

Install Python and [Postgresql](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-16-04).

Then, install `pip3` packages: `pip3 install setuptools nltk numpy pandas scikit-learn --user`.

# Database

Create a new table for the processed tweets:

```sql
CREATE TABLE "public"."tweets" (
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
