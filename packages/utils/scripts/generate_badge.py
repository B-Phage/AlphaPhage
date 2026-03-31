import argparse
import math
import sys

# --- Configuration Constants ---
FONT_FAMILY = "Verdana,Geneva,DejaVu Sans,sans-serif"
FONT_SIZE = 11  # px
HEIGHT = 20  # Total badge height (fixed)
TEXT_PADDING = 6  # px left/right padding for text
LABEL_BASE_WIDTH = 45  # Minimum starting width for the label section


def get_badge_color(coverage: str) -> str:
    """
    Determines the badge color based on the coverage percentage string.
    Matches the logic from the original YAML's badge_color function.
    """
    try:
        cov_float = float(coverage)
        if cov_float >= 95.0:
            return "green"
        elif cov_float >= 85.0:
            return "yellow"
        else:
            return "red"
    except ValueError:
        # Default to red or a safe color if input is invalid
        return "red"


def calculate_text_width(text: str) -> int:
    """
    Estimates the pixel width of the text string using a typical font estimate.
    1 character is roughly 6px wide at 11pt.
    """
    # Round up to the nearest pixel and add padding
    return math.ceil(len(text) * 6.5) + (TEXT_PADDING * 2)


def generate_svg_badge(label: str, message: str, color: str) -> str:
    """
    Generates a complete SVG badge string with auto-calculated widths.
    """
    # 1. Calculate Widths

    # The message is the coverage percentage (e.g., "95.5"). Add '%' for width calculation.
    message_text = f"{message}%"

    # Calculate widths of the two parts
    label_text_width = calculate_text_width(label)
    message_part_width = calculate_text_width(message_text)

    # Ensure label part has a minimum width
    label_width = max(label_text_width, LABEL_BASE_WIDTH)

    # Message part is fixed by its content width
    message_width = message_part_width

    total_width = label_width + message_width

    # 2. Calculate Text Positions (x coordinates for text-anchor="middle")
    label_text_x = label_width / 2
    message_text_x = label_width + (message_width / 2)

    # Text Y position (center of the 20px height)
    text_y = HEIGHT / 2 + 4  # 4px adjustment for typical baseline alignment

    # 3. Generate SVG Template (using the structure from your YAML)
    svg_template = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{HEIGHT}">
    <linearGradient id="b" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
      <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
      <rect width="{total_width}" height="{HEIGHT}" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
      <rect width="{label_width}" height="{HEIGHT}" fill="#555"/>
      <rect x="{label_width}" width="{message_width}" height="{HEIGHT}" fill="{color}"/>
      <rect width="{total_width}" height="{HEIGHT}" fill="url(#b)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{FONT_SIZE}">
      <text x="{label_text_x}" y="{text_y}">{label}</text>
      <text x="{message_text_x}" y="{text_y}">{message_text}</text>
    </g>
</svg>
    """
    return svg_template


def main():
    """Parses command-line arguments and prints the generated SVG to stdout."""
    parser = argparse.ArgumentParser(description="Generate an auto-sized SVG coverage badge.")
    parser.add_argument("label_prefix", help="The static label (e.g., 'coverage').")
    parser.add_argument("branch_name", help="The dynamic branch name (e.g., 'main').")
    parser.add_argument("coverage_value", help="The coverage percentage string (e.g., '95.5').")
    parser.add_argument(
        "--output",
        help="Output file path (e.g., 'badge.svg'). If not provided, prints to stdout.",
        default=None,
    )

    args = parser.parse_args()

    # Combine prefix and branch name for the full label (e.g., "coverage main")
    full_label = f"{args.label_prefix} {args.branch_name}"

    # Determine color
    color = get_badge_color(args.coverage_value)

    svg_content = generate_svg_badge(full_label, args.coverage_value, color)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(svg_content.strip())
            print(f"Successfully generated badge to {args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing to file {args.output}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(svg_content.strip())


if __name__ == "__main__":
    main()
