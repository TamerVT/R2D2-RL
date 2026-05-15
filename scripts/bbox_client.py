import json
import socket
import struct
import cv2


class BBoxServiceClient:
    """
    Client for bbox_service.py

    Protocol:
      send:  uint32_be N + JPEG bytes
      recv:  uint32_be M + UTF-8 JSON bytes (list of [x1,y1,x2,y2])
    """

    def __init__(self, host="127.0.0.1", port=5555, timeout_s=0.2, jpeg_quality=85):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.jpeg_quality = int(jpeg_quality)
        self._sock = None

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def _connect(self):
        if self._sock is not None:
            return
        s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        s.settimeout(self.timeout_s)
        self._sock = s

    @staticmethod
    def _recv_exact(sock, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("bbox service closed connection")
            data += chunk
        return data

    def detect(self, bgr_image):
        """
        Returns: list of (x1,y1,x2,y2) ints in pixel coords of the image you send.
        On failure: returns [] (and resets connection).
        """
        try:
            self._connect()

            ok, buf = cv2.imencode(
                ".jpg",
                bgr_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return []

            payload = buf.tobytes()
            self._sock.sendall(struct.pack("!I", len(payload)) + payload)

            hdr = self._recv_exact(self._sock, 4)
            (m,) = struct.unpack("!I", hdr)
            resp = self._recv_exact(self._sock, m)

            boxes = json.loads(resp.decode("utf-8"))
            # boxes is expected to be list of [x1,y1,x2,y2]
            return [tuple(map(int, b)) for b in boxes]

        except Exception:
            # important: don't stall your control loop on network issues
            self.close()
            return []