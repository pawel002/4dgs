
### Running the rendering

Static room:

```bash
CUDA_VISIBLE_DEVICES=0 /usr/local/blender-5.1.1-linux-x64/blender -b ./room/room-static-object-room.blend -e 150 -P script-static.py
```

Dynamic room:

```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/blender-5.1.1-linux-x64/blender -b ./room/room-dynamic.blend -e 270 -P script-dynamic.py
```