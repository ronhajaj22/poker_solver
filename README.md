# Screenshot Tool

A Python tool for taking screenshots with various options.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Screenshot
Take a screenshot of the entire screen:
```bash
python main.py
```

### Screenshot with Custom Name
```bash
python main.py --output my_screenshot.png
```

### Screenshot with Delay
Take a screenshot after 3 seconds:
```bash
python main.py --delay 3
```

### Region Screenshot
Take a screenshot of a specific region (x=100, y=100, width=500, height=300):
```bash
python main.py --mode region --x 100 --y 100 --width 500 --height 300
```

### Multiple Screenshots
Take 10 screenshots with 5-second intervals:
```bash
python main.py --mode multiple --count 10 --interval 5
```

### Save to Custom Folder
```bash
python main.py --mode multiple --folder my_screenshots
```

## Command Line Options

- `--mode`: Screenshot mode (`single`, `region`, `multiple`)
- `--output` or `-o`: Output filename
- `--delay` or `-d`: Delay before taking screenshot (seconds)
- `--count` or `-c`: Number of screenshots for multiple mode
- `--interval` or `-i`: Interval between screenshots (seconds)
- `--folder` or `-f`: Folder to save screenshots
- `--x`, `--y`, `--width`, `--height`: Coordinates for region screenshots

## Examples

1. **Quick screenshot**: `python main.py`
2. **Screenshot with 5-second delay**: `python main.py --delay 5`
3. **Save as specific file**: `python main.py --output desktop.png`
4. **Region screenshot**: `python main.py --mode region --x 0 --y 0 --width 800 --height 600`
5. **Multiple screenshots**: `python main.py --mode multiple --count 3 --interval 10`

## Notes

- Screenshots are saved as PNG files
- If no output filename is specified, a timestamp-based name is used
- The tool creates folders automatically if they don't exist
- Region screenshots require all four coordinates (x, y, width, height) 