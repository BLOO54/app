"""Picture styles, sizes and quality presets shown in the app."""

STYLES = {
    "none":      ("No style", "{p}"),
    "photo":     ("Photo", "{p}, professional photograph, sharp focus, natural lighting, highly detailed"),
    "cinematic": ("Cinematic", "{p}, cinematic film still, dramatic lighting, shallow depth of field, epic"),
    "anime":     ("Anime", "{p}, anime style, vibrant colors, clean line art, detailed"),
    "cartoon":   ("Cartoon", "{p}, cartoon style, bold outlines, bright colors, fun"),
    "3d":        ("3D render", "{p}, 3d render, octane render, soft lighting, highly detailed, smooth"),
    "game":      ("Game art", "{p}, video game concept art, stylized, vibrant, detailed environment"),
    "fantasy":   ("Fantasy", "{p}, epic fantasy art, magical atmosphere, highly detailed, painterly"),
    "pixel":     ("Pixel art", "{p}, pixel art, 16-bit, retro game sprite style, crisp pixels"),
    "comic":     ("Comic book", "{p}, comic book art, bold ink lines, halftone shading, dynamic"),
    "painting":  ("Oil painting", "{p}, oil painting, visible brush strokes, rich colors, fine art"),
    "logo":      ("Logo / icon", "{p}, clean vector logo icon, simple shapes, flat colors, centered, white background"),
}

ASPECTS = {
    "square":    ("Square", 1024, 1024),
    "portrait":  ("Tall (phone)", 832, 1216),
    "landscape": ("Wide", 1216, 832),
    "wide":      ("Extra wide (wallpaper)", 1344, 768),
}

QUALITY = {"turbo": "Turbo (fastest, needs Speed pack)", "fast": "Fast", "balanced": "Balanced", "best": "Best"}


def apply_style(prompt: str, style: str) -> str:
    _label, template = STYLES.get(style, STYLES["none"])
    return template.replace("{p}", prompt.strip())
