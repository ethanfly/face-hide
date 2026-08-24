from __future__ import annotations

from collections.abc import Callable

from PySide6.QtNetwork import QLocalServer, QLocalSocket

INSTANCE_NAME = "FaceHide.SingleInstance"


def notify_existing(name: str = INSTANCE_NAME, timeout_ms: int = 400) -> bool:
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(timeout_ms):
        socket.close()
        return False
    socket.write(b"raise\n")
    socket.flush()
    socket.waitForBytesWritten(timeout_ms)
    socket.disconnectFromServer()
    socket.close()
    return True


class SingleInstance:
    def __init__(
        self,
        parent=None,
        name: str = INSTANCE_NAME,
        on_activate: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.on_activate = on_activate
        self.server = QLocalServer(parent)
        self.server.newConnection.connect(self._on_connect)
        self._owned = False

    @property
    def owned(self) -> bool:
        return self._owned

    def acquire(self) -> bool:
        if notify_existing(self.name):
            return False
        for _ in range(2):
            QLocalServer.removeServer(self.name)
            if self.server.listen(self.name):
                self._owned = True
                return True
            if notify_existing(self.name):
                return False
        return True

    def close(self) -> None:
        self.server.close()
        if self._owned:
            QLocalServer.removeServer(self.name)
            self._owned = False

    def _on_connect(self) -> None:
        incoming = self.server.nextPendingConnection()
        if incoming is not None:
            incoming.readyRead.connect(incoming.readAll)
        if self.on_activate is not None:
            self.on_activate()
