import asyncio
import json
import os
import pathlib
from aiohttp import web, WSMsgType
from tiago_sim import TiagoSim
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc_server import WebRTCServer

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('WebSocket client connected')
    sim = request.app['sim']
    try:
        while True:
            msg = await ws.receive()
            # print(f'[WS] Received: {msg.data}')  # Debug print
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                # print(f'[WS] Parsed: {data}')  # Debug print
                if data.get('type') == 'control':
                    print(f'[WS] Control command: {data}')
                    sim.step(data)
                elif data.get('type') == 'set_step_size':
                    print(f"[WS] Set step size: {data.get('value')}")
                    sim.step({'command': 'set_step_size', 'value': data.get('value')})
                elif data.get('type') == 'switch_mocap':
                    print('[WS] Switch mocap command received')
                    sim.step({'command': 'switch_mocap'})
                elif data.get('type') == 'set_mocap':
                    sim.current_mocap_index = int(data.get('index', 0))
                    print(f'[WS] Set mocap index to {sim.current_mocap_index}')
            elif msg.type == WSMsgType.ERROR:
                print('WebSocket error:', ws.exception())
            elif msg.type == WSMsgType.CLOSE:
                print('[WS] Connection closed by client')
                break
    except Exception as e:
        print(f'WebSocket loop error: {e}')
    print('WebSocket connection closed')
    return ws

async def index_handler(request):
    index_path = pathlib.Path(__file__).parent.parent / "client" / "index.html"
    return web.FileResponse(index_path)

sim = TiagoSim()
webrtc_server = WebRTCServer(sim)
app = web.Application()
app['sim'] = sim
app.router.add_get('/ws', websocket_handler)
app.router.add_post('/offer', webrtc_server.offer)
app.router.add_get('/', index_handler)
app.router.add_static('/', pathlib.Path(__file__).parent.parent / "client")

if __name__ == '__main__':
    print('Running server at http://localhost:8080')
    web.run_app(app, port=8080)