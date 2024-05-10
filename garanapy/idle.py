import time
import threading

from rich import print as rprint

animation = [
             " [=       ]",
             " [ =      ]",
             " [  =     ]",
             " [   =    ]",
             " [    =   ]",
             " [     =  ]",
             " [      = ]",
             " [       =]",
             " [      = ]",
             " [     =  ]",
             " [    =   ]",
             " [   =    ]",
             " [  =     ]",
             " [ =      ]"
                           ]

class Idle:
    def __init__(self, idle_string, finish_string) -> None:
        self.idle_string   = idle_string
        self.finish_string = finish_string

    def start_idle(self):
        self.done_training = False
        def animate():
            idx = 0
            while not self.done_training:
                rprint(f'{self.idle_string} [bold blue]'+animation[idx % len(animation)], end="\r")
                idx += 1
                time.sleep(0.1)

        t = threading.Thread(target=animate)
        t.start()

    def end_idle(self):
        self.done_training = True
        rprint(f'[green]{self.finish_string}')