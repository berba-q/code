"""
TITLE:          plot comparison of FAO-IFA data
DESCRIPTION:    This scipt implements the multiple_charts_plot module to create charts comparing FAO-IFA data.
VERSION:        v1.0
PURPOSE:        Generate multi-page PDF reports with charts organized by series, plots, pages, and files
"""


# LIBRARIES
import pandas as pd

# Call the multiple_charts_plot module
from multiple_charts_plot import (multiple_line_charts,ChartConfig, ChartStyle,
                                    create_color_assignment, create_line_assignment, 
                                    create_symbol_assignment
                                )

# read already cleaned and processed FAO-IFA data
data = pd.read_csv("data/fao_ifa_2025.csv")

#print(data.head(5))

# Prepare chart configuration
style = ChartStyle.from_data(
    data,
    series_by="d.source",
    colors=["#5792C9", "#F9844A"], #blue FAO, orange IFA
    linetypes=["solid", "dashed"] # solid FAO, dashed IFA
)

# Prepare chart configuration
config = ChartConfig(
    series_by="d.source",
    symbol_by="Flag",
    style=style,
    plots_by="ElementName",
    x_variable="Year",
    y_variable="Value",
    units="Unit",
    plots_per_row=2,
    plots_per_column=2,
    pages_by="Area_Item",
    output_folder="output/",
    file_name="FAO_IFA_compare",
    legend_position='right',
    show_all_years=False,
    show_minor_grid=True,
    marker_size=8,
)

# Generate the plots
result = multiple_line_charts(data, config)

