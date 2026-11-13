# AI Image Studio

Make AI pictures on your own PC: no limits, no account, no cost per picture.
It runs on your NVIDIA graphics card and opens in its own window.

![AI Image Studio](https://github.com/BLOO54/app/releases/latest/download/window.png)

## Install
1. Go to **Releases** (right side of this page) and download `AIImageStudio-Setup-x.x.x.exe`.
2. Double-click it. If Windows says *"Windows protected your PC"*, click **More info → Run anyway**.
3. No administrator rights needed.

## First start (one time only)
The app walks you through it on the **Setup** tab:
1. **Install the AI engine** (about 3 GB). This is the software that lets your graphics card draw.
2. **Download a model** (about 7 GB each):
   - **Juggernaut XL**: realistic photos, people, places, game art, pretty much anything
   - **Animagine XL 4.0**: anime and cartoon style

After that it works offline, as much as you want.

## How to use
- Type what you want to see, pick a **style** (Photo, Anime, Game art, Pixel art...), a **size** and **how many**, then press **Generate** (or Ctrl + Enter).
- Click any picture to see its settings, download it, delete it, or make more like it.
- Everything you make is saved in **Pictures\AI Image Studio**.
- **More options**: things you *don't* want in the picture, a fixed seed (the same seed + the same prompt gives the same picture), and Safe mode.

## Updating
Click **Check for updates** at the bottom of the app. Updates are small and keep your engine, models and pictures.

## Good to know
- On an 8 GB card (like the RTX 4060), each 1024×1024 picture takes roughly 10-20 seconds. The first picture after starting takes longer because the model has to load.
- Closing the window closes the app about a minute later and frees your graphics card.
- Data (settings, log, models) is kept in `%LOCALAPPDATA%\AIImageStudio`. Uninstalling removes it but keeps your pictures.
- Safe mode (on by default) steers away from nudity and gore. Prompts that mix children with sexual content are always refused.
- The models are free under the CreativeML Open RAIL++-M license. Read their pages on Hugging Face for what's allowed.

## Building
Every push to `main` is built on GitHub's Windows machines (see `.github/workflows/build-windows.yml`).
It's tested there, including installing the engine and drawing a test picture, and then published as a release.
