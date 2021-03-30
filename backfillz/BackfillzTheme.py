from typing import List


class BackfillzTheme:
    """Backfillz visualisation settings."""

    def __init__(
        self,
        name: str,
        text_family: str,
        text_font: float,
        text_font_colour: str,
        text_cex_title: float,
        text_cex_main: float,
        text_cex_axis: float,
        text_col_title: str,
        text_col_main: str,
        text_col_axis: str,
        bg_colour: str,
        mg_colour: str,
        fg_colour: str,
        alpha: float,
        palette: List[str]
    ):
        """Construct theme object."""
        self.name = name
        self.text_family = text_family
        self.text_font = text_font
        self.text_font_colour = text_font_colour
        self.text_cex_title = text_cex_title
        self.text_cex_main = text_cex_main
        self.text_cex_axis = text_cex_axis
        self.text_col_title = text_col_title
        self.text_col_main = text_col_main
        self.text_col_axis = text_col_axis
        self.bg_colour = bg_colour
        self.mg_colour = mg_colour
        self.fg_colour = fg_colour
        self.alpha = alpha
        self.palette = palette


default: BackfillzTheme = BackfillzTheme(
    name="default",
    text_family="sans",
    text_font=1,
    text_font_colour="black",
    text_cex_title=1.5,
    text_cex_main=1,
    text_cex_axis=0.8,
    text_col_title="grey10",
    text_col_main="grey60",
    text_col_axis="grey40",
    bg_colour="white",
    mg_colour="grey50",
    fg_colour="black",
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
    text_col_title="grey10",
    text_col_main="grey60",
    text_col_axis="grey40",
    bg_colour="#002B36",
    mg_colour="grey50",
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
    text_font_colour="grey20",
    text_cex_title=1.5,
    text_cex_main=1,
    text_cex_axis=0.6,
    text_col_title="grey10",
    text_col_main="grey60",
    text_col_axis="grey40",
    bg_colour="grey98",
    mg_colour="grey90",
    fg_colour="grey40",
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
    text_font_colour="grey90",
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
