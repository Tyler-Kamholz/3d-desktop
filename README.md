# 3d-desktop

Turns your monitor into a window onto a 3D scene: either your own photo
(converted to layered depth) or a built-in grid room. Your webcam tracks your head and the
view is re-projected from where your eyes are, so the orange grid room and the furry
"meatball" look like they sit *behind* the glass. It's a browser take on a
head-coupled-perspective TouchDesigner demo (subsurface-ish fur shader included).

## Run it

It's one file, `index.html`, with no build step. Three.js and the MediaPipe face tracker load from a CDN, so
you need to be online.

- **Easiest:** open `index.html` in Chrome or Edge (double-click it).
- **If the camera won't start from a file**, serve the folder locally instead:
  ```sh
  python3 -m http.server 8000
  # then open http://localhost:8000
  ```

Then:

1. Click **Start head tracking** and allow camera access.
2. Set **Screen size** to your display's diagonal in inches. This matters for the illusion.
3. Press **F** for fullscreen and **H** to hide the panel.
4. Sit about an arm's length away and move your head around.

Without a webcam, the view follows your mouse.

## Use your own photo

`tools/make_scene.py` turns a photo into a 3D scene. It works best with a clear subject
in front of a background.

```sh
pip install torch transformers pillow opencv-python-headless numpy
python tools/make_scene.py my-photo.jpg --focus 0.55 --standalone my-photo.standalone.html
```

The script:

1. Estimates depth with Depth Anything V2.
2. Cuts out the nearest subject.
3. Fills in what's behind the subject with LaMa inpainting, so moving your head reveals plausible
   background instead of a smear.
4. Writes `scene.js`, which `index.html` loads automatically. With `--standalone`, it also writes a
   single HTML file you can double-click.

`--focus` sets the vertical point of interest: 0 is the top of the photo and 1 is the bottom.
`--threshold` sets how near something has to be to count as foreground (default 0.5).

`scene.js` and `*.standalone.html` are git-ignored so personal photos stay out of the repo.

With a photo loaded, the panel adds:

- **Scene:** switch between the photo and the grid room.
- **Photo depth:** how far behind the screen the farthest part of the photo sits. Higher means more
  parallax, but you'll see the edges sooner.
- **Framing:** which part of a tall photo fills a wide screen.

## Tuning

| Setting | What it does |
| --- | --- |
| Screen size | Physical size of the display, used to convert pixels to centimetres. |
| Webcam field of view | Horizontal FOV of your webcam. If moving toward or away from the screen feels too strong or too weak, adjust this. |
| Parallax boost | Exaggerates sideways and up/down head movement. 1 is physically correct. |
| Flip left/right | Use this if the scene moves the wrong way when you move your head (some webcams mirror their feed). |

The illusion is strongest with one eye closed, or when filmed with a phone, because both
eyes can't share one viewpoint.

## How it works

- **Head tracking:** MediaPipe Face Landmarker finds your irises. The distance between them in
  pixels gives your distance from the screen (people's pupils are about 6.3 cm apart), and their position in the
  frame gives you x/y. A One Euro filter smooths out jitter.
- **Off-axis projection:** the camera sits at your eye position, and its frustum is skewed
  so its edges always pass through the physical corners of your screen. That's what makes the screen act as a window.
- **Fur:** 48 shells of a lumpy sphere, each cut down to strand cross-sections in the
  fragment shader, plus line-segment stray hairs.
- **"Subsurface":** wrap lighting that lets red wrap further around the shadow edge than
  green or blue, plus back-lit translucency at the silhouette.
