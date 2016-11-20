import io


def lazy_read_file(filename):
    with io.open(filename, buffering = 1) as file:
        while file.readable():
            yield file.readline()
