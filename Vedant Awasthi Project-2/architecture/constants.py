# ──────────────────────────────────────────────
# constants.py — shared constants for MK Motion Control
# ──────────────────────────────────────────────

MODE_CURSOR  = "CURSOR"
MODE_CNN     = "CNN"
MODE_DRAWING = "DRAWING"
MODE_SANDBOX = "SANDBOX"

CNN_IMG_SIZE          = 64      # skeleton image size (px)
MIN_SAMPLES_PER_CLASS = 15
MAX_SAMPLES_PER_CLASS = 200
DIVERSITY_THRESHOLD   = 0.025   # min landmark diff to accept a capture frame
CNN_CONFIDENCE_THRESHOLD = 0.75 # min confidence to fire a bound action

DRAW_COLORS = [
    (0,   0,   255),  # Red
    (0,   127, 255),  # Orange
    (0,   255, 255),  # Yellow
    (0,   255, 0),    # Green
    (255, 255, 0),    # Cyan
    (255, 0,   0),    # Blue
    (255, 0,   127),  # Purple
    (255, 255, 255),  # White
]

MP_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# action_id → (display label, category, debounce_seconds)
# category: 'cursor' | 'draw' | 'both' | 'none'
ACTION_CATALOGUE = {
    'move_cursor':   ('Move Cursor',   'cursor', 0.0),
    'left_click':    ('Left Click',    'cursor', 0.3),
    'right_click':   ('Right Click',   'cursor', 0.4),
    'scroll_up':     ('Scroll Up',     'cursor', 0.08),
    'scroll_down':   ('Scroll Down',   'cursor', 0.08),
    'vol_up':        ('Volume Up',     'cursor', 0.1),
    'vol_down':      ('Volume Down',   'cursor', 0.1),
    'task_view':     ('Task View',     'cursor', 1.0),
    'draw':          ('Draw',          'draw',   0.0),
    'erase':         ('Erase',         'draw',   0.0),
    'clear':         ('Clear Canvas',  'draw',   1.5),
    'color_next':    ('Next Color',    'draw',   0.4),
    'brush_bigger':  ('Brush Bigger',  'draw',   0.15),
    'brush_smaller': ('Brush Smaller', 'draw',   0.15),
    'none':          ('(no action)',   'none',   0.0),
}

# Keys active in the binding panel → action_id
BINDING_KEYS = {
    ord('1'): 'move_cursor',
    ord('2'): 'left_click',
    ord('3'): 'right_click',
    ord('4'): 'scroll_up',
    ord('5'): 'scroll_down',
    ord('6'): 'vol_up',
    ord('7'): 'vol_down',
    ord('8'): 'task_view',
    ord('a'): 'draw',
    ord('d'): 'erase',
    ord('f'): 'clear',
    ord('g'): 'color_next',
    ord('h'): 'brush_bigger',
    ord('j'): 'brush_smaller',
    ord('0'): 'none',
}