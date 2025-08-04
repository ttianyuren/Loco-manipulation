from aiortc import VideoStreamTrack
from av import VideoFrame
import asyncio

class MuJoCoVideoStream(VideoStreamTrack):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
    async def recv(self):
        # This is for WebRTC, but for WebSocket image streaming, use get_frame_base64
        pts, time_base = await self.next_timestamp()
        frame = self.sim.get_frame()
        video_frame = VideoFrame.from_ndarray(frame, format='bgr24')
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    async def get_ws_frame(self):
        # For WebSocket streaming to <img src="...">
        frame_data = self.sim.get_frame_base64()
        mocap_index = self.sim.current_mocap_index
        return {
            'type': 'frame',
            'data': frame_data,
            'mocap_index': mocap_index
        }
