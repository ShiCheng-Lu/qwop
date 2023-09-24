from multiprocessing import Process
import http.server
import socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="game", **kwargs)

    def log_message(self, format, *args):
        return


def _host_game():
    with socketserver.TCPServer(("", 8000), Handler) as httpd:
        httpd.serve_forever()

def start():
    global game_host

    game_host = Process(target=_host_game)
    game_host.start()

def end():
    global game_host
    # pass
    game_host.kill()
