#!/usr/bin/python3
def filtering_data(a_dictionary):
    return list(
        map(
            lambda x: x['name'],
            filter(lambda x: x['salary'] > 10000, a_dictionary)
        )
    )
