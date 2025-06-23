#!/usr/bin/env python3


import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np


def create_data(n):
    data = []
    np_data = np.random.normal(size=(n, 2))

    for p in np_data:
        data.append({'x': p[0], 'y': p[1]})
    return data


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            data = create_data(200000)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

if __name__ == '__main__':
    HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
