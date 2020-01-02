# highlight-extraction

usage: `python app.py low-res high-res outname sec`
ex: `python app.py 360p.mp4 720p.mp4 highlight.mp4 180`

low resolution is for model, because model will resize the resolution to 320x180, so lower resolution is desirable

high resolution is for the final video, we reassemble the highlight from the high resolution one
