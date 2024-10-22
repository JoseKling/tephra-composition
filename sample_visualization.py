import marimo

__generated_with = "0.8.9"
app = marimo.App(width="full")


@app.cell
def __():
    import io
    import re
    import functions
    import marimo                      as mo
    import pandas                      as pd
    import numpy                       as np
    import matplotlib.pyplot           as plt
    import seaborn                     as sns
    from sklearn.preprocessing         import LabelEncoder
    from sklearn.decomposition         import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    return LDA, LabelEncoder, PCA, functions, io, mo, np, pd, plt, re, sns


@app.cell
def __(mo):
    f = mo.ui.file(filetypes=[".csv"], multiple=False)
    return f,


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
        samples_column = mo.ui.dropdown(options=_non_features + ["None"],  value="None", label="**Groups column**: ")
    return df_temp, features, samples_column, x_axis, y_axis


@app.cell
def __(mo):
    plot_scatter  = mo.ui.switch(value=True, label="Scatter")
    plot_kde      = mo.ui.switch(value=False, label="KDE")
    return plot_kde, plot_scatter


@app.cell
def __(df_temp, f, mo, np, re, samples_column, sns):
    if f.name() is not None:
        if samples_column.value != "None":
            _col_vals = df_temp[samples_column.value].unique()
            _all_samples = [re.match(r"-.+-", s).group(0) for s in _col_vals]
            palette = {_all_samples[i]: sns.color_palette()[i % len(sns.color_palette())] for i in (np.arange(len(_all_samples)))}
            samples_uis = mo.ui.dictionary({sample: mo.ui.checkbox(value=False, label=sample) for sample in _all_samples})
    return palette, samples_uis


@app.cell
def __(f, mo, samples_column, samples_uis, x_axis, y_axis):
    if f.name() is not None:
        if samples_column.value == "None":
            _output = mo.md(f"""
                Choose the elements to be plotted and the column with the different groups (sources) to be differentiated.  
                If there are elements in the table that are not showing below, check the file for typos.
                {mo.vstack([x_axis, y_axis, samples_column], justify="center")}
            """)
        else:
            _output = mo.md(f"""
                Choose the elements to be plotted and the column with the different groups (sources) to be differentiated.  
                If there are elements in the table that are not showing below, check the file for typos.
                {mo.vstack([x_axis, y_axis, samples_column], justify="center")}
                **Sources to plot**
                {mo.hstack(samples_uis.values())}
            """)
            samples = [s[0] for s in samples_uis.items() if s[1].value]
    else:
        samples = []
        _output = mo.md("")
    _output
    return samples,


@app.cell
def __():
    #if not()
    return


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
