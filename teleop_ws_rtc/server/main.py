import asyncio
import json
import os
import pathlib
from aiohttp import web, WSMsgType
from tiago_sim_clean import TiagoSim
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc_server import WebRTCServer

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Generate unique connection ID for tracking
    connection_id = id(ws)
    print(f'[WS] Client connected (ID: {connection_id})')
    sim = request.app['sim']
    
    # Performance optimization: reduce message processing overhead
    ping_count = 0
    
    try:
        while True:
            msg = await ws.receive()
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                # Fast path for ping messages (highest priority)
                if data.get('type') == 'ping':
                    ping_count += 1
                    pong_response = {
                        'type': 'pong',
                        'timestamp': data.get('timestamp'),
                        'server_time': asyncio.get_event_loop().time() * 1000
                    }
                    await ws.send_str(json.dumps(pong_response))
                    # Only log every 50th ping to reduce overhead and show connection ID
                    if ping_count % 50 == 0:
                        print(f'[WS] Connection {connection_id}: {ping_count} pings processed')
                    continue
                
                # Process other message types
                if data.get('type') == 'control':
                    # Reduce console spam - only log every few control messages
                    if hasattr(sim, '_control_count'):
                        sim._control_count += 1
                    else:
                        sim._control_count = 1
                    
                    if sim._control_count % 20 == 0:  # Log every 20th command
                        print(f'[WS] Control command #{sim._control_count}: {data.get("command")}')
                    
                    # Track command latency if client timestamp is provided
                    client_timestamp = data.get('client_timestamp')
                    if client_timestamp:
                        sim.track_command_latency(data.get('command', 'unknown'), client_timestamp, ws)
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
                print(f'[WS] Connection {connection_id} closed by client')
                break
    except Exception as e:
        print(f'[WS] Connection {connection_id} error: {e}')
    print(f'[WS] Connection {connection_id} terminated')
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