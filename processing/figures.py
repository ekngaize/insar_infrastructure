import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import TwoSlopeNorm

def plot_map(gdf, column, vmin, vcenter, vmax, column_unit, vmin_label, vcenter_label):
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    norm = TwoSlopeNorm(
        vmin=vmin,
        vcenter=vcenter,
        vmax=vmax
    )

    gdf.plot(
        column=column,
        cmap="RdYlGn",
        linewidth=0.5,
        edgecolor="black",
        legend=False,
        norm=norm,
        ax=ax,
        alpha=0.4
    )

    for idx, row in gdf.iterrows():
        x, y = row.geometry.centroid.coords[0]
        val = row[column]

        label = f"{val:.1f} {column_unit}"
        if val < vmin_label:
            label += "\n⬇︎"
        elif val < vcenter_label:
            label += "\n➡"
        else:
            label += "\n⬆"

        ax.text(
            x, y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            weight="bold"
        )

    cx.add_basemap(
        ax,
        source=cx.providers.GeoportailFrance.orthos
    )

    ax.set_axis_off()

    ax.set_title(
        f'{column} in {gdf['orbite'].unique()[0]} orbite',
        fontsize=13,
        fontweight='bold',
        loc='left'
    )

    plt.savefig(f'output_data/{column}_{gdf['orbite'].unique()[0]}.png')
    print(f'Write {column} in {gdf['orbite'].unique()[0]} orbite.')
