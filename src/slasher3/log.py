from typing import Callable

class Pen:
    hd = "\033["

    colors = {
        'red': f"{hd}31m",
        'yellow': f"{hd}93m",
        'blue': f"{hd}36m"
    }
    tail = f"{hd}00m"

    def __getattr__(self, __name: str) -> Callable[[str], str]:
        if __name in self.colors:
            return lambda color: f"{self.colors[__name]}{color}{self.tail}"

pen = Pen()

class Logger:
    pen : Pen
    def __init__(self, pen : Pen):
        self.pen = pen

    def info(self, where : str, msg : str) -> None:
        self.log(self.pen.blue('INFO'), where, f"{msg}")
    
    def warn(self, where : str, msg : str) -> None:
        self.log(self.pen.yellow('WARN'), where, f"{msg}")

    def error(self, where : str, msg : str, exception : str = None) -> None:
        why = self.pen.red('ERROR')
        if exception:
            self.log(why, where, f"{msg} with error {self.pen.red(exception)}!")
        else:
            self.log(why, where, msg)

    def log(self, why: str, where : str, msg : str) -> None:
        print(f"[{why}] {where}: {msg}")

logger = Logger(pen)