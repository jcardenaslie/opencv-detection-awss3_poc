import threading
import time

def loop1_10():
    for i in range(1, 11):
        # time.sleep(1)
        print('1: {}'.format(i))

threading.Thread(target=loop1_10).start()

for i in range(10):
    print('2: {}'.format(i))