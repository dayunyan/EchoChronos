import asyncio
from typing import Callable
import websockets


class WebSocketManager:
    def __init__(self, host: str, port: int, handler: Callable):
        self.host = host
        self.port = port
        self.handler = handler
        self.connected_clients = []  # 用于存储已连接的客户端

    async def init_socket(self):
        async with websockets.serve(
            self.handle_connection, self.host, self.port, ping_interval=None
        ):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # 保持服务器运行

    async def handle_connection(self, websocket, path):
        self.connected_clients.append(websocket)
        try:
            async for message in websocket:
                print(f"Received message: {message}")
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosedError:
            print("A client has disconnected.")
        finally:
            self.connected_clients.remove(websocket)

    async def handle_message(self, message):
        print(f"Handling message: {message}")
        response = self.handler(message)
        self.send(response)

    def send(self, data):
        for client in self.connected_clients:
            asyncio.run_coroutine_threadsafe(
                client.send(data), asyncio.get_event_loop()
            )

    def recv(self):
        raise NotImplementedError(
            "The recv method should be implemented in a more specific way "
            "depending on your requirements. For example, you could "
            "implement it to wait for and return a specific message "
            "from a client."
        )

    async def disconnect_all_clients(self):
        for client in self.connected_clients:
            await client.close()
        self.connected_clients.clear()

    def __del__(self):
        asyncio.run(asyncio.ensure_future(self.disconnect_all_clients()))
        print("WebSocket server closed.", flush=True)
