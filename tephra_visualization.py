import marimo

__generated_with = "0.8.9"
app = marimo.App(width="full")


@app.cell
def __():
    import io
    import functions
    import marimo                      as mo
    import pandas                      as pd
    import numpy                       as np
    import matplotlib.pyplot           as plt
    import seaborn                     as sns
    from sklearn.preprocessing         import LabelEncoder
    from sklearn.decomposition         import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    return LDA, LabelEncoder, PCA, functions, io, mo, np, pd, plt, sns


@app.cell
def __(mo):
    f = mo.ui.file(filetypes=[".csv"], multiple=False)
    return f,


@app.cell
def __(mo):
    mo.md(
        """
        # Visualization of tephra composition

        This is a tephra composition visualization app.  
        The composition data must be provided as a table in csv format (it is just a regular excel table, but when saving, go to 'save as' and choose the csv format).
        """
    )
    return


@app.cell
def __(f, mo):
    mo.md(f"""
        Upload the csv file.
        {f}
    """)
    return


@app.cell
def __(all_features, f, functions, io, mo, np, pd):
    if f.name() is not None:
        _df            = pd.read_csv(io.StringIO(f.contents().decode()))
        features       = list(set(_df.columns.str.strip()) & set(all_features))
        df_temp        = functions.clean_data(_df, features)
        _non_features  = list(set(df_temp.select_dtypes(exclude=np.number).columns.tolist()).difference(set(all_features)))
        x_axis         = mo.ui.radio(options=features, value="SiO2", label="**x-axis**: ", inline=True)
        y_axis         = mo.ui.radio(options=features, value="TAS",  label="**y-axis**: ", inline=True)
        sources_column = mo.ui.dropdown(options=_non_features + ["None"],  value="None", label="**Groups column**: ")
    return df_temp, features, sources_column, x_axis, y_axis


@app.cell
def __(mo):
    plot_scatter  = mo.ui.switch(value=True, label="Scatter")
    plot_kde      = mo.ui.switch(value=False, label="Density")
    return plot_kde, plot_scatter


@app.cell
def __(df_temp, f, mo, np, sns, sources_column):
    if f.name() is not None:
        if sources_column.value != "None":
            _all_sources = df_temp[sources_column.value].unique()
            _all_sources = [x if isinstance(x, str) else None for x in _all_sources]
            palette = {_all_sources[i]: sns.color_palette()[i % len(sns.color_palette())] for i in (np.arange(len(_all_sources)))}
            sources_uis = mo.ui.dictionary({source: mo.ui.checkbox(value=False, label=source) for source in _all_sources})
    return palette, sources_uis


@app.cell
def __(f, mo, sources_column, sources_uis, x_axis, y_axis):
    if f.name() is not None:
        if sources_column.value == "None":
            sources = []
            _output = mo.md(f"""
                Choose the elements to be plotted and the column with the different groups (sources) to be differentiated.  
                If there are elements in the table that are not showing below, check the file for typos.
                {mo.vstack([x_axis, y_axis, sources_column], justify="center")}
                
            """)
        else:        
            _output = mo.md(f"""
                Choose the elements to be plotted. If there are elements in the table that are not showing below, check the file for typos.
                {mo.vstack([x_axis, y_axis, sources_column], justify="center")}
                **Sources to plot**
                {mo.hstack(sources_uis.values())}
            """)
            sources = [s[0] for s in sources_uis.items() if s[1].value]
    else:
        sources = []
        _output = mo.md("")
    _output
    return sources,


@app.cell
def __(
    df_temp,
    f,
    features,
    mo,
    plot_kde,
    plot_scatter,
    sources,
    sources_column,
):
    if f.name() is not None:
        if sources_column.value == "None":
            df = df_temp
        else:
            df = df_temp.loc[df_temp[sources_column.value].isin(sources), features + [sources_column.value]] 
    if f.name() is not None:
        _output = mo.md(f"""
            Choose the type of plot (Can select both): {mo.hstack([plot_scatter, plot_kde], justify="start", gap=10)}
        """)
    else:
        _output = mo.md("")
    _output
    return df,


@app.cell
def __(
    df,
    f,
    kde_bw,
    palette,
    plot_kde,
    plot_scatter,
    plot_x_lim,
    plot_y_lim,
    plt,
    scatter_size,
    sns,
    sources_column,
    x_axis,
    y_axis,
):
    if f.name() is not None:
        # If KDE plot is selected
        _fig, _ax = plt.subplots()
        if plot_kde.value:
            if sources_column.value != "None":
                sns.kdeplot(ax=_ax, data=df, x=x_axis.value, y=y_axis.value, hue=sources_column.value,
                            palette=palette, alpha=0.5, fill=True, bw_adjust=kde_bw.value, levels=4, common_norm=False)
            else:
                sns.kdeplot(ax=_ax, data=df, x=x_axis.value, y=y_axis.value,
                            alpha=0.5, fill=True, levels=4, common_norm=False, label="Density", bw_adjust=kde_bw.value)
        _ax.set_prop_cycle(None)
        # if scatter plot is selected
        if plot_scatter.value:
            if sources_column.value != "None":
                sns.scatterplot(ax=_ax, data=df, x=x_axis.value, y=y_axis.value,
                                palette=palette, s=scatter_size.value, hue=sources_column.value)
            else:
                sns.scatterplot(ax=_ax, data=df, x=x_axis.value, y=y_axis.value,
                                s=scatter_size.value, label="Samples")
        if plot_scatter.value or plot_kde.value:
            sns.move_legend(_ax, "center left", bbox_to_anchor=(1,0.5))
        _ax.set_xlim(plot_x_lim.value)
        _ax.set_ylim(plot_y_lim.value)
    else:
        _fig = None
    _fig
    return


@app.cell
def __(f, mo):
    plot_customize = mo.ui.checkbox(value=False)
    if f.name() is not None:
        _output = mo.md(f"""
            Do you want to customize the plot? {plot_customize}        
        """)
    else:
        _output = mo.md("")
    _output
    return plot_customize,


@app.cell
def __(df, f, mo, plot_customize, sources, x_axis, y_axis):
    if f.name() is not None:
        if sources:
            _x_min = min(df[x_axis.value])
            _x_max = max(df[x_axis.value])
            _y_min = min(df[y_axis.value])
            _y_max = max(df[y_axis.value])
            plot_x_lim = mo.ui.range_slider(start=_x_min-(0.1*_x_max), stop=_x_max*1.1, step=0.1, value=[_x_min-(0.1*_x_max), _x_max*1.1], full_width=True, show_value=True, label="**x-axis limits:** ")
            plot_y_lim = mo.ui.range_slider(start=_y_min-(0.1*_y_max), stop=_y_max*1.1, step=0.1, value=[_y_min-(0.1*_y_max), _y_max*1.1], full_width=True, show_value=True, label="**y-axis limits:** ")
    if plot_customize.value:
        _output = mo.md(f"""
        Use the axis limits below to zoom in/out
        {mo.vstack([plot_x_lim, plot_y_lim])}""")
    else:
        _output = mo.md("")
    _output
    return plot_x_lim, plot_y_lim


@app.cell
def __(mo, plot_customize, plot_scatter):
    scatter_size = mo.ui.slider(start=1, stop=30, step=1, value=12, label="**Scatter plot point size:** ")
    if plot_customize.value & plot_scatter.value:
        _output = mo.md(f"{scatter_size}")
    else:
        _output = mo.md("")
    _output
    return scatter_size,


@app.cell
def __(mo, plot_customize, plot_kde):
    kde_bw = mo.ui.slider(start=0.1, stop=2, step=0.1, value=1, label="**Density plot bandwidt:** ")
    if plot_customize.value & plot_kde.value:
        _output = mo.md(f"{kde_bw}")
    else:
        _output = mo.md("")
    _output
    return kde_bw,


@app.cell
def __(mo):
    plot_pca = mo.ui.switch(value=False, label="PCA")
    plot_lda = mo.ui.switch(value=False, label="LDA")
    return plot_lda, plot_pca


@app.cell
def __(f, mo, plot_lda, plot_pca, sources, sources_column):
    if f.name() is not None:
        if len(sources) == 0:
            _output = None
        elif (sources_column.value == "None") | (len(sources) <= 2):
            _output = mo.md(f"""
               It is possible to apply [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to the data and possibly have a better visualization.  
                {plot_pca}
            """)
        else:
            _output = mo.md(f"""
                It is also possible to visualize the data with [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) or with [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).  
                Choose below if you want to try any of these methods.
                {mo.hstack([plot_pca, plot_lda], justify="start", gap=10)}
            """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(mo, plot_pca):
    if plot_pca.value:
        _output = mo.md("""
            ### PCA
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(PCA, df, features, np, plot_pca, plt):
    if plot_pca.value:
        pca = PCA().fit(df[features].values)
        _x_ticks = np.array(range(1, len(pca.explained_variance_) + 1, round((len(pca.explained_variance_) + 1) / 5)))
        _fig, _ax = plt.subplots(figsize=(6,2))
        plt.bar(range(1, len(pca.explained_variance_)+1), pca.explained_variance_ratio_)
        plt.ylabel("Explained\nvariance")
        plt.xlabel("Component")
        _ax.set_xticks(_x_ticks)
    else:
        _fig = None
    _fig
    return pca,


@app.cell
def __(mo):
    pca_scatter  = mo.ui.switch(value=True, label="Scatter")
    pca_kde  = mo.ui.switch(value=False, label="KDE")
    return pca_kde, pca_scatter


@app.cell
def __(mo, pca_kde, pca_scatter, plot_pca):
    if plot_pca.value:
        _output = mo.md(f"""
            Choose the type of plot (Can select both): {mo.hstack([pca_scatter, pca_kde], justify="start", gap=10)}
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(df, features, pca, pd, plot_pca, sources_column):
    if plot_pca.value:
        _X_transf = pca.transform(df[features].values)
        df_pca = pd.DataFrame(_X_transf, columns= ["Component "+str(i) for i in range(1, len(features) + 1)])
        if sources_column.value != "None":
            df_pca.insert(0, sources_column.value, df[sources_column.value].values)
    return df_pca,


@app.cell
def __(
    df_pca,
    palette,
    pca_kde,
    pca_kde_bw,
    pca_scatter,
    pca_scatter_size,
    pca_x_lim,
    pca_y_lim,
    plot_pca,
    plt,
    sns,
    sources_column,
):
    if plot_pca.value:
        # If KDE plot is selected
        _fig, _ax = plt.subplots()
        if pca_kde.value:
            if sources_column.value != "None":
                sns.kdeplot(ax=_ax, data=df_pca, x="Component 1", y="Component 2", hue=sources_column.value,
                            palette=palette, alpha=0.5, fill=True, bw_adjust=pca_kde_bw.value, levels=4, common_norm=False)
            else:
                sns.kdeplot(ax=_ax, data=df_pca, x="Component 1", y="Component 2",
                            alpha=0.5, fill=True, levels=4, common_norm=False, label="Density", bw_adjust=pca_kde_bw.value)
        _ax.set_prop_cycle(None)
        # if scatter plot is selected
        if pca_scatter.value:
            if sources_column.value != "None":
                sns.scatterplot(ax=_ax, data=df_pca, x="Component 1", y="Component 2",
                                palette=palette, s=pca_scatter_size.value, hue=sources_column.value)
            else:
                sns.scatterplot(ax=_ax, data=df_pca, x="Component 1", y="Component 2",
                                s=pca_scatter_size.value, label="Samples")
        sns.move_legend(_ax, "center left", bbox_to_anchor=(1,0.5))
        _ax.set_xlim(pca_x_lim.value)
        _ax.set_ylim(pca_y_lim.value)
    else:
        _fig = None
    _fig
    return


@app.cell
def __(mo):
    pca_customize = mo.ui.checkbox(value=False)
    return pca_customize,


@app.cell
def __(mo, pca_customize, plot_pca):
    if plot_pca.value:
        _output = mo.md(f"""
            Do you want to customize the plot? {pca_customize}        
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(df_pca, mo, pca_customize, plot_pca):
    if plot_pca.value:
        _x_min = min(df_pca["Component 1"].values)
        _x_max = max(df_pca["Component 1"].values)
        _y_min = min(df_pca["Component 2"].values)
        _y_max = max(df_pca["Component 2"].values)
        pca_x_lim = mo.ui.range_slider(start=_x_min-(0.1*_x_max), stop=_x_max*1.1, step=0.1, value=[_x_min-(0.1*_x_max), _x_max*1.1], full_width=True, show_value=True, label="**x-axis limits:** ")
        pca_y_lim = mo.ui.range_slider(start=_y_min-(0.1*_y_max), stop=_y_max*1.1, step=0.1, value=[_y_min-(0.1*_y_max), _y_max*1.1], full_width=True, show_value=True, label="**y-axis limits:** ")
    if pca_customize.value:
        _output = mo.md(f"""
        Use the axis limits below to zoom in/out
        {mo.vstack([pca_x_lim, pca_y_lim])}""")
    else:
        _output = mo.md("")
    _output
    return pca_x_lim, pca_y_lim


@app.cell
def __(mo, pca_customize, pca_scatter, plot_pca):
    if plot_pca.value:
        pca_scatter_size = mo.ui.slider(start=1, stop=30, step=1, value=12, label="**Scatter plot point size:** ")
    if pca_customize.value & pca_scatter.value:
        _output = mo.md(f"{pca_scatter_size}")
    else:
        _output = mo.md("")
    _output
    return pca_scatter_size,


@app.cell
def __(mo, pca_customize, pca_kde, plot_pca):
    if plot_pca.value:
        pca_kde_bw = mo.ui.slider(start=0.1, stop=2, step=0.1, value=1, label="**Density plot bandwidt:** ")
    if pca_customize.value & pca_kde.value:
        _output = mo.md(f"{pca_kde_bw}")
    else:
        _output = mo.md("")
    _output
    return pca_kde_bw,


@app.cell
def __(mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        _output = mo.md("""
            ### LDA
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(LDA, df, features, np, plot_lda, plt, sources, sources_column):
    if (plot_lda.value) & (len(sources) > 2):
        lda = LDA().fit(df[features].values, df[sources_column.value].values)
        _x_ticks = np.array(np.arange(1, len(lda.explained_variance_ratio_) + 1, round((len(lda.explained_variance_ratio_) + 1))))
        _fig, _ax = plt.subplots(figsize=(6,2))
        plt.bar(range(1, len(lda.explained_variance_ratio_)+1), lda.explained_variance_ratio_)
        plt.ylabel("Explained\nvariance")
        plt.xlabel("Component")
        _ax.set_xticks(_x_ticks)
    else:
        _fig = None
    _fig
    return lda,


@app.cell
def __(mo):
    lda_scatter  = mo.ui.switch(value=True, label="Scatter")
    lda_kde  = mo.ui.switch(value=False, label="KDE")
    return lda_kde, lda_scatter


@app.cell
def __(lda_kde, lda_scatter, mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        _output = mo.md(f"""
            Choose the type of plot (Can select both): {mo.hstack([lda_scatter, lda_kde], justify="start", gap=10)}
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(df, features, lda, np, pd, plot_lda, sources, sources_column):
    if (plot_lda.value) & (len(sources) > 2):
        _X_transf = lda.transform(df[features].values)
        df_lda = pd.DataFrame(_X_transf, columns= ["Component "+str(i) for i in np.arange(1, len(lda.explained_variance_ratio_) + 1)])
        if sources_column.value != "None":
            df_lda.insert(0, sources_column.value, df[sources_column.value].values)
    return df_lda,


@app.cell
def __(
    df_lda,
    lda_kde,
    lda_kde_bw,
    lda_scatter,
    lda_scatter_size,
    lda_x_lim,
    lda_y_lim,
    palette,
    plot_lda,
    plt,
    sns,
    sources,
    sources_column,
):
    if (plot_lda.value) & (len(sources) > 2):
        # If KDE plot is selected
        _fig, _ax = plt.subplots()
        if lda_kde.value:
            if sources_column.value != "None":
                sns.kdeplot(ax=_ax, data=df_lda, x="Component 1", y="Component 2", hue=sources_column.value,
                            palette=palette, alpha=0.5, fill=True, bw_adjust=lda_kde_bw.value, levels=4, common_norm=False)
            else:
                sns.kdeplot(ax=_ax, data=df_lda, x="Component 1", y="Component 2",
                            alpha=0.5, fill=True, levels=4, common_norm=False, label="Density", bw_adjust=lda_kde_bw.value)
        _ax.set_prop_cycle(None)
        # if scatter plot is selected
        if lda_scatter.value:
            if sources_column.value != "None":
                sns.scatterplot(ax=_ax, data=df_lda, x="Component 1", y="Component 2",
                                palette=palette, s=lda_scatter_size.value, hue=sources_column.value)
            else:
                sns.scatterplot(ax=_ax, data=df_lda, x="Component 1".value, y="Component 2".value,
                                s=lda_scatter_size.value, label="Samples")
        sns.move_legend(_ax, "center left", bbox_to_anchor=(1,0.5))
        _ax.set_xlim(lda_x_lim.value)
        _ax.set_ylim(lda_y_lim.value)
    else:
        _fig = None
    _fig
    return


@app.cell
def __(mo):
    lda_customize = mo.ui.checkbox(value=False)
    return lda_customize,


@app.cell
def __(lda_customize, mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        _output = mo.md(f"""
            Do you want to customize the plot? {lda_customize}        
        """)
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def __(df_lda, lda_customize, mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        _x_min = min(df_lda["Component 1"].values)
        _x_max = max(df_lda["Component 1"].values)
        _y_min = min(df_lda["Component 2"].values)
        _y_max = max(df_lda["Component 2"].values)
        lda_x_lim = mo.ui.range_slider(start=_x_min-(0.1*_x_max), stop=_x_max*1.1, step=0.1, value=[_x_min-(0.1*_x_max), _x_max*1.1], full_width=True, show_value=True, label="**x-axis limits:** ")
        lda_y_lim = mo.ui.range_slider(start=_y_min-(0.1*_y_max), stop=_y_max*1.1, step=0.1, value=[_y_min-(0.1*_y_max), _y_max*1.1], full_width=True, show_value=True, label="**y-axis limits:** ")
    if lda_customize.value:
        _output = mo.md(f"""
        Use the axis limits below to zoom in/out
        {mo.vstack([lda_x_lim, lda_y_lim])}""")
    else:
        _output = mo.md("")
    _output
    return lda_x_lim, lda_y_lim


@app.cell
def __(lda_customize, lda_scatter, mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        lda_scatter_size = mo.ui.slider(start=1, stop=30, step=1, value=12, label="**Scatter plot point size:** ")
    if lda_customize.value & lda_scatter.value:
        _output = mo.md(f"{lda_scatter_size}")
    else:
        _output = mo.md("")
    _output
    return lda_scatter_size,


@app.cell
def __(lda_customize, lda_kde, mo, plot_lda, sources):
    if (plot_lda.value) & (len(sources) > 2):
        lda_kde_bw = mo.ui.slider(start=0.1, stop=2, step=0.1, value=1, label="**Density plot bandwidt:** ")
    if lda_customize.value & lda_kde.value:
        _output = mo.md(f"{lda_kde_bw}")
    else:
        _output = mo.md("")
    _output
    return lda_kde_bw,


@app.cell
def __():
    all_features = ["Na2O", "K2O", "FeO", "SiO2", "TiO2", "MgO", "CaO", "MnO", "Al2O3", "P2O5", "TAS", "SI/K", "Mg/Ca", "Fe/Mg",
        "Fe/Ti", "Fe/Si", "Fe/Ca", "Ti/K20", "SI/TI", "Li7", "Mg24", "Al27", "P31", "Ca43", "Sc45",
        "Ti47", "V51", "Cr52", "Mn55", "Co59", "Cu63", "Zn66", "Ga69", "Rb85", "Sr88", "Y89", "Zr90", "Nb93", "Sn118",
        "Cs133", "Ba137", "La139", "Ce140", "Pr141", "Nd146", "Sm147", "Eu153", "Gd157", "Tb159", "Dy163", "Ho165", "Er166", "Tm169",
        "Yb172", "Lu175", "Hf178", "Ta181", "Pb208", "Th232", "U238", "La/Th", "La/Yb", "Rb/La", "La/Sm", "Ce/Yb",
        "Zr/Cs", "Pb/Nd", "Li/Y", "Nb/Rb", "Ba/Ce", "Ce/Pb", "U/Th", "Ba/La", "U/Pb", "Ba/Rb",
        "Nb/Ta", "Ba/Th", "Th/Ta", "Ba/Nb", "Nb/Th", "Rb/Hf", "Rb/Nd", "Rb/Sr", "Ba/Zr", "Ti/Zr",
        "Zr/Nb", "Dy/Lu", "La/Nb", "Sm/Yb", "Ta/U", "Nb/Zr", "Ce/Eu", "Ce/Hf", "Sm/Nd", "Sm/Zr", "Lu/Hf", "K/La",
        "Rb/Th", "Nb/Ti", "Zr/Hf", "Nb/U", "Rb/Cs", "Ce/U", "Sr/Nd", "U/La"]
    return all_features,


if __name__ == "__main__":
    app.run()
