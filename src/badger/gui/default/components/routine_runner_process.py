# https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
# 0. import libraries that do not use multithreading:
import os, sys, time
from multiprocessing import Process, Pipe


class RoutineRunnerProcess(Process):

    def __init__(self):
        super().__init__()
        # 2. create interprocess communication primitives and shared resources used by the current multiprocess:
        self.front_pipe, self.back_pipe = Pipe()

    # BACKEND

    def run(self):
        # 4. import libraries that use multithreading:
        # from SomeLibrary import Analyzer
        # self.analyzer = Analyzer()
        ...
        # 5. if you use asyncio, remember to create a new event loop
        print("Routine Runner Process: run")
        self.active = True
        while self.active:
            self.active = self.listenFront__()
        print("bye from multiprocess!")

    def listenFront__(self):
        message = self.back_pipe.recv()
        if message == "stop":
            return False
        elif message == "ping":
            self.ping__()
            return True
        else:
            print("listenFront__ : unknown message", message)
            return True

    def ping__(self):
        print("backend: got ping from frontend")
        self.back_pipe.send("pong")

    # FRONTEND

    def ping(self):
        self.front_pipe.send("ping")
        msg = self.front_pipe.recv()
        print("frontend: got a message from backend:", msg)

    def stop(self):
        self.front_pipe.send("stop")
        self.join()


if __name__ == "__main__":
    p = RoutineRunnerProcess()
    # 3. fork (=create multiprocesses)
    p.start()  # this performs fork & start p.run() "on the other side" of the fork
    print("front: sleepin 5 secs")
    time.sleep(5)
    p.ping()
    print("front: sleepin 5 secs")
    time.sleep(5)
    p.stop()
