# Attributes of every tweet

Just to make sure the dataset is consistent, I run this command to check what attributes each tweet contains.

    cat tweets2009-06.txt | tail -n +2 | cut -c1 | sort | uniq -c | tail -n +2
    
The result is satisfying.

    18572084 T
    18572084 U
    18572084 W
    
That means that each tweet in the dataset contains exactly 3 attributes:

- Timestamp
- User name
- Tweet content
