import io


def lazy_read_file(filename):
    with io.open(filename, buffering = 1, encoding = 'utf-8') as file:
        while file.readable():
            yield file.readline()

