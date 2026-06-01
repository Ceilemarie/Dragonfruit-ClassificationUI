import io
import time
import sys
from PIL import Image

URL = 'http://127.0.0.1:5000/analyze'

# create a simple test image
img = Image.new('RGB', (224,224), (120,200,80))
for x in range(60,160):
    for y in range(60,160):
        img.putpixel((x,y), (200,30,30))

# prepare multipart form-data without requests
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
buf = io.BytesIO()
buf.write(('--%s\r\n' % boundary).encode())
buf.write(b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n')
buf.write(b'Content-Type: image/png\r\n\r\n')
img.save(buf, format='PNG')
buf.write(b'\r\n--%s--\r\n' % boundary.encode())
body = buf.getvalue()

import http.client

# retry until server responds
for i in range(15):
    try:
        conn = http.client.HTTPConnection('127.0.0.1', 5000, timeout=5)
        conn.request('POST', '/analyze', body=body, headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        })
        resp = conn.getresponse()
        data = resp.read()
        print('STATUS', resp.status)
        print(data.decode('utf-8', errors='replace'))
        conn.close()
        break
    except Exception as e:
        print('Attempt', i+1, 'failed:', e)
        time.sleep(1)
else:
    print('Server did not respond; ensure it is running', file=sys.stderr)
    sys.exit(2)
