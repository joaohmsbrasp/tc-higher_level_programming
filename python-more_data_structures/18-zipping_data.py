#!/usr/bin/python3
def zipping_data(list1, list2):
    for product, price in zip(list1, list2):
        print(f"{product}: R$ {price:.2f}")