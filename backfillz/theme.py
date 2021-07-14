from dataclasses import dataclass
from typing import List


@dataclass
class BackfillzTheme:
    """Backfillz visualisation settings. Colours are hex strings, without alphas."""

    name: str
    text_family: str
    text_font: float
    text_font_colour: str
    text_cex_title: float
    text_cex_main: float
    text_cex_axis: float
    text_col_title: str
    text_col_main: str
    text_col_axis: str
    bg_colour: str
    mg_colour: str
    fg_colour: str
    alpha: float
    palette: List[str]


default: BackfillzTheme = BackfillzTheme(
    name="default",
    text_family="sans",
    text_font=1,
    text_font_colour="#000000",
    text_cex_title=1.5,
    text_cex_main=1,
    text_cex_axis=0.8,
    text_col_title="#1a1a1a",
    text_col_main="#999999",
    text_col_axis="#666666",
    bg_colour="#ffffff",
    mg_colour="#7f7f7f",
    fg_colour="#000000",
    alpha=0.7,
    palette=[
        "#FF0000",
        "#0000FF",
        "#FF00FF",
        "#800000",
        "#000080",
        "#FF6347"
    ]
)

solarized_dark: BackfillzTheme = BackfillzTheme(
    name="solarized_dark",
    text_family="mono",
    text_font=1,
    text_font_colour="#2AA198",
    text_cex_title=2,
    text_cex_main=1,
    text_cex_axis=0.8,
    text_col_title="#1a1a1a",
    text_col_main="#999999",
    text_col_axis="#666666",
    bg_colour="#002B36",
    mg_colour="#7f7f7f",
    fg_colour="#93A1A1",
    alpha=0.7,
    palette=[
        "#657B83",
        "#D30102",
        "#D33682",
        "#859900",
        "#93A1A1",
        "#268BD2"
    ]
)

demo_1: BackfillzTheme = BackfillzTheme(
    name="demo 1",
    text_family="mono",
    text_font=1,
    text_font_colour="#333333",
    text_cex_title=1.5,
    text_cex_main=1,
    text_cex_axis=0.6,
    text_col_title="#1a1a1a",
    text_col_main="#999999",
    text_col_axis="#666666",
    bg_colour="#fafafa",
    mg_colour="#e5e5e5",
    fg_colour="#666666",
    alpha=0.8,
    palette=[
        "#A3C96D",
        "#DDCF1E",
        "#8E4D91",
        "#003B24",
        "#912B2F",
        "#7C6EAC"
    ]
)

demo_2: BackfillzTheme = BackfillzTheme(
    name="demo 2",
    text_family="sans",
    text_font=1,
    text_font_colour="#e5e5e5",
    text_cex_title=1.5,
    text_cex_main=1,
    text_cex_axis=0.6,
    text_col_title="#F2EEE7",
    text_col_main="#F2EEE7",
    text_col_axis="#F2EEE7",
    bg_colour="#313C3F",
    mg_colour="#313C3F",  # adjustcolor("#313C3F", red.f = 1.8, green.f = 1.8, blue.f = 1.8),
    fg_colour="#F2EEE7",
    alpha=0.8,
    palette=[
        "#EEE436",
        "#00AEC7",
        "#C73475",
        "#7FC5D3",
        "#7EB627",
        "#F29530"
    ]
)
